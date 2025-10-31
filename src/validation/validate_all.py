import os
import pathlib

import numpy as np
import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.validation.utils import (open_results,
                                  get_auc_curve_data,
                                  plot_individual_auc_curve,
                                  plot_multi_auc_curves)
from src.algorithms.time_of_day.likelihood_to_risk import get_event_inds



""""
Script to generate and save AUC plots for all patients with results from the time_of_day algorithm.
Saves individual plots in ./results/auc_curves/{patient_id}.png
and a combined plot in ./results/auc_curves/all.png
"""

"""
INPUT ALGO_NAME AND USE_RANDOM_OUTPUTS TO CONFIGURE THE SCRIPT BEHAVIOR
"""

ALGO_NAME = input("Enter the algorithm name (use the name of the folder, e.g. time_of_day or moving_average): ") or "time_of_day"
USE_SIGNIFICANCE_TESTING = False

def run_auc_plots(patient_ids, algo_name: str):

    results_data = open_results(patient_ids, algo_name = algo_name)

    # tranform results data into y_true and likelihood arrays
    y_trues = []
    likelihoods = []
    eligible_ids = []
    for patient_id in patient_ids:
        result = results_data[patient_id]
        likelihood = np.array(result['likelihoods'])
        if likelihood.size == 0:
            continue
        times = np.array(result['likelihood_times'])
        events = np.array(result['event_times'])
        event_inds = get_event_inds(times, events)
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
                                  os.path.join("results", "auc_curves", f"{patient_id}_{algo_name}.png"), use_random_outputs=USE_SIGNIFICANCE_TESTING)
    plot_multi_auc_curves(
        fpr, tpr, roc_auc, n_patients,
        [i.replace(f'_{ALGO_NAME}_results', '')
         for i in eligible_ids], os.path.join("results", "auc_curves", f"all_{algo_name}.png"), use_random_outputs=USE_SIGNIFICANCE_TESTING)


if __name__ == "__main__":
    # Open all existing results files
    patient_ids = [
        file_name.replace(f"_{ALGO_NAME}_results.json", '') for file_name in os.listdir('results')
        if file_name.endswith(f"_{ALGO_NAME}_results.json")
    ]

    os.makedirs(os.path.join("results", "auc_curves"), exist_ok=True)
    run_auc_plots(patient_ids, ALGO_NAME)
