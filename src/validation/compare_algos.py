import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


## python -m src run --algos time_of_day,moving_average --ids NV_1,NV_2,NV_3,NV_4,NV_5,NV_6,NV_7,NV_8,NV_9,NV_10,NV_11,NV_12,NV_13,NV_14,NV_15 --outputs prospective -mp -p


if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.validation.utils import open_results

ALGO_1 = input("Enter the first algorithm name (use the name of the folder, e.g. time_of_day or moving_average): ") or "time_of_day"
ALGO_2 = input("Enter the second algorithm name (use the name of the folder, e.g. time_of_day or moving_average): ") or "moving_average"


from src.validation.utils import (open_results,
                                  get_auc_curve_data,
                                  plot_individual_auc_curve,
                                  plot_multi_auc_curves)
from src.algorithms.time_of_day.likelihood_to_risk import get_event_inds

def run_auc_plots(patient_ids, algo_name, algo_type = False, auc_scores_only = False):

    auc_scores = {}
    event_count = {}
    testing_days = {}
    results_data = open_results(patient_ids, algo_name)

    if algo_name == "time_of_day":
        shorten = True
        results_ma = open_results(patient_ids, "moving_average")
    else:
        shorten = False

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

                ## recalculate likleihoods:
                if len(result["likelihoods"]) / 24 == len(likelihood):
                    # calculate the likelihood array as the average of the result['likelihoods'] every 24 hours
                    likelihood = np.array(result['likelihoods']).reshape(-1, 24).mean(axis=1)
                    if times[0] != result['likelihood_times'][0]:
                        times = np.array(result['likelihood_times']).reshape(-1, 24).min(axis=1)

                if shorten:
                    first_time = results_ma[patient_id]['daily_likelihood_times'][0]
                    last_time = results_ma[patient_id]['daily_likelihood_times'][-1]
                    filter_times = (times >= first_time) & (times <= last_time)
                    times = times[filter_times]
                    likelihood = likelihood[filter_times]
                    events = events[(events >= first_time) & (events <= last_time)]

            elif algo_type == "hourly":
                likelihood = np.array(result['likelihoods'])
                times = np.array(result['likelihood_times'])
                if shorten:
                    first_time = results_ma[patient_id]['likelihood_times'][0]
                    last_time = results_ma[patient_id]['likelihood_times'][-1]
                    filter_times = (times >= first_time) & (times <= last_time)
                    times = times[filter_times]
                    likelihood = likelihood[filter_times]
                    events = events[(events >= first_time) & (events <= last_time)]

            if likelihood.size == 0:
                continue

            event_inds = get_event_inds(times, events, algo_type)
            y_true = np.zeros(len(likelihood))
            y_true[event_inds] = 1

            likelihoods.append(likelihood)
            y_trues.append(y_true)
            eligible_ids.append(patient_id)

            # ignore patients with less than 5 events or less than 30 days of testing data
            if (np.sum(y_true) < 5) or (len(likelihood) / 24 < 30 if algo_type == "hourly" else len(likelihood) < 30):
                continue

            # auc score
            event_count[patient_id] = np.sum(y_true)
            testing_days[patient_id] = len(likelihood) / 24 if algo_type == "hourly" else len(likelihood)
            auc_scores[patient_id] = roc_auc_score(y_true, likelihood)

        except Exception as e:
            print(f"Skipping patient {patient_id} due to error: {e}")
            continue

    if auc_scores_only:
        return auc_scores, event_count, testing_days

    y_trues = np.array(y_trues, dtype=object)
    likelihoods = np.array(likelihoods, dtype=object)
    n_patients = len(eligible_ids)

    # Plot AUC curves
    fpr, tpr, roc_auc = get_auc_curve_data(y_trues, likelihoods, n_patients)
    for i, patient_id in enumerate(eligible_ids):
        plot_individual_auc_curve(patient_id, fpr[i], tpr[i], roc_auc[i],
                                  os.path.join("results", "auc_curves", f"{patient_id}_{algo_name}_{algo_type}.png"), False)
    plot_multi_auc_curves(
        fpr, tpr, roc_auc, n_patients,
        eligible_ids, os.path.join("results", "auc_curves", f"all_{algo_name}_{algo_type}.png"), False)

    return auc_scores, event_count, testing_days

if __name__ == "__main__":

    os.makedirs(os.path.join("results", "auc_curves"), exist_ok=True)

    df = pd.DataFrame()
    for algo_type in ['hourly', 'daily']:
        for algo_name in [ALGO_1, ALGO_2]:
            patient_ids = [
                file_name.replace(f'_{algo_name}_results.json', '') for file_name in os.listdir('results')
                if file_name.endswith(f'_{algo_name}_results.json')
            ]

            auc_scores, event_count, testing_days = run_auc_plots(patient_ids, algo_name, algo_type, auc_scores_only=True)
            if "patient_id" not in df.columns:
                df['patient_id'] = list(auc_scores.keys())

            if algo_type == 'hourly' and algo_name == ALGO_1:
                df[f"testing_events"] = [event_count[p_id] if p_id in event_count else None for p_id in df['patient_id']]
                df[f"testing_days"] = [testing_days[p_id] if p_id in testing_days else None for p_id in df['patient_id']]
                df[f"seizure_frequency"] = df["testing_events"] / df["testing_days"]
            
            df[f"auc_{algo_name}_{algo_type}"] = [auc_scores[p_id] if p_id in auc_scores else None for p_id in df['patient_id']]

    df[f"hourly_{ALGO_1}>{ALGO_2}"] = df[f"auc_{ALGO_1}_hourly"] > df[f"auc_{ALGO_2}_hourly"]
    df[f"daily_{ALGO_1}>{ALGO_2}"] = df[f"auc_{ALGO_1}_daily"] > df[f"auc_{ALGO_2}_daily"]
    df.to_csv(os.path.join("results", "compare_algos_auc_scores.csv"))