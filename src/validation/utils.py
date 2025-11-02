from datetime import datetime
import os
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray as Array
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import dates
from itertools import cycle
from src.algorithms.time_of_day.likelihood_to_risk import get_event_inds
from src.utils import read_json

def convert_timestamps(inp_list):
    return [datetime.fromtimestamp(t / 1000) for t in inp_list]


def set_date_axis():
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.MonthLocator())
    # ax.xaxis.set_major_locator(dates.DayLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %y'))


def read_json_results(file_path: str) -> Any:
    """Opens json files of results and returns None if file does not exist."""
    try:
        return read_json(file_path)
    except FileNotFoundError:
        return None


# Open all existing results files
def open_results(patient_ids: Sequence[str], algo_name) -> Dict[str, Any]:
    """
    Opens all results for given patient ids

    Parameters
    -------
    patient_ids: sequence of str
        list of patient ids

    Returns
    -------
    results: dict of results
        keys are strings of patient ids, values are patient results
    """
    results = {}
    for patient_id in patient_ids:
        result = read_json_results(os.path.join("results", f'{patient_id}_{algo_name}_pseudoprospective_outputs.json'))
        if result is None:
            print(f'{patient_id} had no result data.')
            continue
        results[patient_id] = result
    return results


def get_auc_curve_data(
    y_true: Array, likelihood: Array, n_patients: int
) -> Tuple[Dict[Union[str, int], Array], Dict[Union[str, int], Array], Array]:
    """
    Gathers all data necessary to plot auc curves

    Parameters
    -------
    y_true: array of y_true for each patient
        dimensions correspond to number of IDs
    likelihood: array of likelihoods for each patient
        dimensions correspond to number of IDS
    n_patients: number of IDs

    Returns
    -------
    fpr: array of float
        fpr values for patients
    tpr: array of float
        tpr values for patients
    roc_auc: array of float
        auc scores for patients
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_patients):
        fpr[i], tpr[i], _ = roc_curve(y_true[i], likelihood[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("Average AUC score:", np.mean([i for i in roc_auc.values()]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.concatenate(y_true),
                                              np.concatenate(likelihood))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_patients)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_patients):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_patients

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def auc_random_outputs(patient_id: str) -> List[float]:
    """Opens random outputs for given patient_id and returns auc scores"""

    auc_scores = []
    if not os.path.exists(os.path.join('results', 'randomized_outputs')):
        raise FileNotFoundError('No significance testing was done on this cohort. Please run the algorithm with the --sig_test tag to use this feature.')


    for path in os.listdir(os.path.join('results', 'randomized_outputs')):

        if path.startswith(patient_id):
            result = read_json(os.path.join('results', 'randomized_outputs', path))

            likelihood = np.array(result['likelihoods'])
            times = np.array(result['likelihood_times'])
            events = np.array(result['event_times'])
            event_inds = get_event_inds(times, events)

            y_true = np.zeros(len(likelihood))
            y_true[event_inds] = 1

            fpr, tpr, _ = roc_curve(y_true, likelihood)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)

    return np.percentile(np.array(auc_scores), 95)


def plot_individual_auc_curve(patient_id: str, fpr: Array, tpr: Array,
                              roc_auc: float, save_fig: str, use_random_outputs: bool = True) -> None:
    """
    Plots an individual's auc curve and saves it as save_fig

    Parameters
    -------
    patient_id: str
        patient id
    fpr: array of float
        fpr values
    tpr: array of float
        tpr values
    roc_auc: float
        auc score
    save_fig: str
        file path + file name for saving figure
    """

    if use_random_outputs:
        random_auc = auc_random_outputs(patient_id)
        sig = '*' if roc_auc >= random_auc else ''
    else:
        sig = ''

    # plot individual curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label=f"ROC curve (area = {roc_auc:0.2f}{sig})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.axis('square')
    plt.savefig(save_fig, dpi=300)
    plt.close()


def plot_multi_auc_curves(fpr: Array, tpr: Array, roc_auc: Array,
                          n_patients: int, patient_ids: List[str],
                          save_fig: str, use_random_outputs: bool = True) -> None:
    """
    Plots mulitple auc curves on one plot and saves it as save_fig

    Parameters
    -------
    fpr: array of float
        fpr values for patients
    tpr: array of float
        tpr values for patients
    roc_auc: array of float
        auc scores for patients
    n_patients: int
        number of IDs
    save_fig: str
        file path + file name for saving figure
    """
    # Plot all ROC curves
    fig = plt.figure()
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-avg (area = {roc_auc['micro']:0.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-avg (area = {roc_auc['macro']:0.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle([
        "aqua", "darkorange", "cornflowerblue", "red", "salmon", "lightgreen",
        "darkgreen", "lightseagreen", "darkblue", "purple"
    ])
    for i, color in zip(range(n_patients), colors):

        if use_random_outputs:
            random_auc = auc_random_outputs(patient_ids[i])
            sig = '*' if roc_auc[i] >= random_auc else ''
        else:
            sig = ''
        plt.plot(fpr[i],
                 tpr[i],
                 color=color,
                 lw=lw,
                 label=f"P{patient_ids[i]}") # (area = {roc_auc[i]:0.2f}{sig})")

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.axis('square')
    fig.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.close()
