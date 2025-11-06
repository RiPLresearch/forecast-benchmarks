import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.constants import PATHS
from src.validation.utils import (open_results,
                                  get_auc_curve_data,
                                  plot_individual_auc_curve,
                                  plot_multi_auc_curves)
from src.algorithms.time_of_day.likelihood_to_risk import get_event_inds



""""
Script to generate and save AUC plots for all patients with results from the time_of_day algorithm.
Saves individual plots in ./results/performance_metrics/{patient_id}.png
and a combined plot in ./results/performance_metrics/all.png
"""

"""
INPUT ALGO_NAME AND USE_RANDOM_OUTPUTS TO CONFIGURE THE SCRIPT BEHAVIOR
"""

# ANSI escape code for red text
RED = "\033[31m"
RESET = "\033[0m"

ALGO_NAME = input(f"{RED}Enter the algorithm name (use the name of the folder, e.g. time_of_day or moving_average): {RESET}") or "time_of_day"
TIMESCALE = input(f"{RED}Is the algorithm hourly or daily? Enter 'hourly' or 'daily' (default is 'hourly' if you press enter): {RESET}") or "hourly"

USE_SIGNIFICANCE_TESTING = False

def run_auc_plots(patient_ids, algo_name: str, storage_folder: str = ""):

    results_data = open_results(patient_ids, algo_name = algo_name)

    # tranform results data into y_true and likelihood arrays
    y_trues = []
    likelihoods = []
    eligible_ids = []
    for patient_id in patient_ids:
        result = results_data[patient_id]
        if TIMESCALE == "hourly":
            likelihood = np.array(result['likelihoods'])
            if likelihood.size == 0:
                continue
            # Normalize likelihoods to be in range [0, 1]
            if any(likelihood > 1):
                likelihood = likelihood / np.max(likelihood)
            times = np.array(result['likelihood_times'])
            events = np.array(result['event_times'])
            event_inds = get_event_inds(times, events, algo_type = "hourly")
        
        elif TIMESCALE == "daily":
            likelihood = np.array(result['daily_likelihoods'])
            if likelihood.size == 0:
                continue
            # Normalize likelihoods to be in range [0, 1]
            if any(likelihood > 1):
                likelihood = likelihood / np.max(likelihood)
            times = np.array(result['daily_likelihood_times'])
            events = np.array(result['event_times'])
            event_inds = get_event_inds(times, events, algo_type = "daily")

        y_true = np.zeros(len(likelihood))
        y_true[event_inds] = 1

        likelihoods.append(likelihood)
        y_trues.append(y_true)
        eligible_ids.append(patient_id)

    y_trues = np.array(y_trues, dtype=object)
    likelihoods = np.array(likelihoods, dtype=object)
    n_patients = len(eligible_ids)

    # Plot AUC curves
    fpr, tpr, roc_auc = get_auc_curve_data(y_trues, likelihoods, n_patients)
    for i, patient_id in enumerate(eligible_ids):
        plot_individual_auc_curve(patient_id, fpr[i], tpr[i], roc_auc[i],
                                  os.path.join(storage_folder, f"{patient_id}_{algo_name}.png"), use_random_outputs=USE_SIGNIFICANCE_TESTING)
    plot_multi_auc_curves(
        fpr, tpr, roc_auc, n_patients,
        [i.replace(f'_{ALGO_NAME}_pseudoprospective_outputs', '')
         for i in eligible_ids], os.path.join(storage_folder, f"all_{algo_name}.png"), use_random_outputs=USE_SIGNIFICANCE_TESTING)

    rows = []
    for i, patient_id in enumerate(eligible_ids):
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_trues[i], likelihoods[i])
        pr_auc = auc(recall, precision)
        # Brier
        brier = brier_score_loss(y_trues[i], likelihoods[i])
        rows.append({
            "patient_id": patient_id,
            "roc_auc": float(roc_auc[i]),
            "pr_auc": float(pr_auc),
            "brier": float(brier)
        })

    pd.DataFrame(rows).to_csv(os.path.join(storage_folder, f"performance_scores_{algo_name}.csv"), index=False)

if __name__ == "__main__":
    # Open all existing results files
    patient_ids = [
        file_name.replace(f"_{ALGO_NAME}_pseudoprospective_outputs.json", '') for file_name in os.listdir('results')
        if file_name.endswith(f"_{ALGO_NAME}_pseudoprospective_outputs.json")
    ]

    storage_folder = save_fig = PATHS.results_path("performance_metrics")

    os.makedirs(storage_folder, exist_ok=True)
    run_auc_plots(patient_ids, ALGO_NAME, storage_folder=storage_folder)
