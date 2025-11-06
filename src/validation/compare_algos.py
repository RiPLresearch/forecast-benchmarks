import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.constants import PATHS
from src.validation.utils import open_results

# ANSI escape codes
RED = "\033[31m"
RESET = "\033[0m"

ALGO_1 = input(f"{RED}Enter the first algorithm name (use the name of the folder, e.g. time_of_day or moving_average): {RESET}") or "time_of_day"
ALGO_2 = input(f"{RED}Enter the second algorithm name (use the name of the folder, e.g. time_of_day or moving_average): {RESET}") or "moving_average"


from src.validation.utils import (open_results,
                                  get_auc_curve_data,
                                  plot_individual_auc_curve,
                                  plot_multi_auc_curves)
from src.algorithms.time_of_day.likelihood_to_risk import get_event_inds

def run_auc_plots(patient_ids, algo_name, algo_type = False, auc_scores_only = False, storage_folder = ""):

    auc_scores = {}
    pr_auc_scores = {}
    brier_scores = {} 
    event_count = {}
    testing_days = {}
    results_data = open_results(patient_ids, algo_name)

    # tranform results data into y_true and likelihood arrays
    y_trues = []
    likelihoods = []
    eligible_ids = []
    for patient_id in patient_ids:

        try:
            result = results_data[patient_id]
            events = np.array(result['event_times'])

            if algo_type == "daily":
                likelihood = np.array(result['daily_likelihoods'])
                times = np.array(result['daily_likelihood_times'])

            elif algo_type == "hourly":
                likelihood = np.array(result['likelihoods'])
                times = np.array(result['likelihood_times'])

            if likelihood.size == 0:
                continue
            # Normalize likelihoods to be in range [0, 1]
            if any(likelihood > 1):
                likelihood = likelihood / np.max(likelihood)

            if algo_type == "hourly":
                event_inds = get_event_inds(times, events, algo_type, algo_type == "hourly")
            else:
                event_inds = get_event_inds(times, events, algo_type, algo_type == "daily")

            y_true = np.zeros(len(likelihood))
            y_true[event_inds] = 1

            likelihoods.append(likelihood)
            y_trues.append(y_true)
            eligible_ids.append(patient_id)

            # ignore patients with less than 5 events or less than 30 days of testing data
            if (np.sum(y_true) < 5) or (len(likelihood) / 24 < 30 if algo_type == "hourly" else len(likelihood) < 30):
                continue

            # AUC-ROC score
            event_count[patient_id] = np.sum(y_true)
            testing_days[patient_id] = len(likelihood) / 24 if algo_type == "hourly" else len(likelihood)
            auc_scores[patient_id] = roc_auc_score(y_true, likelihood)

            # AUPRC (PR AUC): use precision_recall_curve then integrate
            precision, recall, _ = precision_recall_curve(y_true, likelihood)
            pr_auc_scores[patient_id] = auc(recall, precision)

            # Brier score
            brier_scores[patient_id] = brier_score_loss(y_true, likelihood)

        except Exception as e:
            print(f"Skipping patient {patient_id} due to error: {e}")
            continue

    if auc_scores_only:
        return auc_scores, event_count, testing_days, pr_auc_scores, brier_scores

    y_trues = np.array(y_trues, dtype=object)
    likelihoods = np.array(likelihoods, dtype=object)
    n_patients = len(eligible_ids)

    # Plot AUC curves
    fpr, tpr, roc_auc = get_auc_curve_data(y_trues, likelihoods, n_patients)
    for i, patient_id in enumerate(eligible_ids):
        plot_individual_auc_curve(patient_id, fpr[i], tpr[i], roc_auc[i],
                                  os.path.join(storage_folder, f"{patient_id}_{algo_name}_{algo_type}.png"), False)
    plot_multi_auc_curves(
        fpr, tpr, roc_auc, n_patients,
        eligible_ids, os.path.join(storage_folder, f"all_{algo_name}_{algo_type}.png"), False)

    return auc_scores, event_count, testing_days

if __name__ == "__main__":

    storage_folder = PATHS.results_path("performance_metrics")

    os.makedirs(storage_folder, exist_ok=True)

    df = pd.DataFrame()
    for algo_type in ['hourly', 'daily']:
        for algo_name in [ALGO_1, ALGO_2]:
            patient_ids = [
                file_name.replace(f'_{algo_name}_pseudoprospective_outputs.json', '') for file_name in os.listdir('results')
                if file_name.endswith(f'_{algo_name}_pseudoprospective_outputs.json')
            ]

            auc_scores, event_count, testing_days, pr_auc_scores, brier_scores = run_auc_plots(patient_ids, algo_name, algo_type, auc_scores_only=True, storage_folder=storage_folder)
            if "patient_id" not in df.columns:
                df['patient_id'] = list(auc_scores.keys())


            if algo_type == 'hourly' and algo_name == ALGO_1:
                df[f"testing_events"] = [event_count[p_id] if p_id in event_count else None for p_id in df['patient_id']]
                df[f"testing_days"] = [testing_days[p_id] if p_id in testing_days else None for p_id in df['patient_id']]
                df[f"seizure_frequency"] = df["testing_events"] / df["testing_days"]
            
            # add metrics to dataframe
            df[f"auc_{algo_name}_{algo_type}"] = [auc_scores[p_id] if p_id in auc_scores else None for p_id in df['patient_id']]
            df[f"pr_auc_{algo_name}_{algo_type}"] = [pr_auc_scores[p_id] if p_id in pr_auc_scores else None for p_id in df['patient_id']]
            df[f"brier_score_{algo_name}_{algo_type}"] = [brier_scores[p_id] if p_id in brier_scores else None for p_id in df['patient_id']]

    df[f"hourly_{ALGO_1}>{ALGO_2}"] = df[f"auc_{ALGO_1}_hourly"] > df[f"auc_{ALGO_2}_hourly"]
    df[f"daily_{ALGO_1}>{ALGO_2}"] = df[f"auc_{ALGO_1}_daily"] > df[f"auc_{ALGO_2}_daily"]
    df.to_csv(os.path.join(storage_folder, f"compare_performance_{ALGO_1}_{ALGO_2}.csv"))
