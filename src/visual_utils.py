from datetime import datetime, timedelta
import os
from matplotlib import dates, pyplot as plt
import numpy as np

from src.algorithms.time_of_day.likelihood_to_risk import get_risk_levels
from src.data import RiskOutput


def matching_timestamp(event, likelihood_dates):

    try:
        return likelihood_dates.index(event.replace(minute=0, second=0,
                                             microsecond=0))
    except:
        return None

def prospective_likelihood_plot(patient_id: str, outputs: RiskOutput,
                                output_path: str, algo_name: str) -> None:
    '''
    Generates a prospective likelihood plot
    '''

    likelihood_times = outputs.likelihood_times
    likelihoods = np.array(outputs.likelihoods)
    med_cutoff = np.array(outputs.medium_thresholds_past)
    high_cutoff = np.array(outputs.high_thresholds_past)
    event_times = np.array(outputs.event_times)

    # Convert UNIX timestamps to datetime
    convert_timestamps = lambda inp_list: [
        datetime.fromtimestamp(t / 1000) for t in inp_list
    ]
    # Dates
    likelihood_dates = convert_timestamps(likelihood_times)
    event_dates = convert_timestamps(event_times)

    # Find hightest and lowest likelihoods
    max_likelihood_idxs = np.where(np.array(likelihoods) == 1)[0]
    i = 0.95
    while not max_likelihood_idxs.any() and i > 0:
        max_likelihood_idxs = np.where(np.array(likelihoods) >= i)[0]
        i -= 0.05

    min_likelihood_idxs = np.where(np.array(likelihoods) == 0)[0]
    j = 0.05
    while not min_likelihood_idxs.any() and j < 1:
        min_likelihood_idxs = np.where(np.array(likelihoods) <= j)[0]
        j += 0.05

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(likelihood_dates, likelihoods, 'grey')

    # plot seizures
    inds = [matching_timestamp(event, likelihood_dates)
        for event in event_dates]
    inds = [i for i in inds if i is not None]

    low_events = np.array(event_dates)[np.where(
        (likelihoods[inds] < med_cutoff[inds]))]
    med_events = np.array(event_dates)[np.where(
        (likelihoods[inds] >= med_cutoff[inds])
        & (likelihoods[inds] < high_cutoff[inds]))]
    high_events = np.array(event_dates)[np.where(
        (likelihoods[inds] >= high_cutoff[inds]))]

    for events, col in zip([low_events, med_events, high_events],
                           ['green', 'orange', 'red']):
        plt.plot(events, [1] * len(events),
                 col,
                 linestyle='',
                 marker='v',
                 markersize=6)

    # plot labels / ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %y'))
    ax.set_ylabel('Likelihood', fontsize=10, fontweight='bold')

    # Figure settings
    plt.setp(ax,
             xlim=(np.min(likelihood_dates), np.max(likelihood_dates)),
             ylim=(0, 1.05))
    ax.set_yticks((0, 1))
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)

    # add future forecast lines (likelihoods and scores)
    plt.plot(likelihood_dates, med_cutoff, color='orange', linewidth=1.5)
    plt.plot(likelihood_dates, high_cutoff, color='red', linewidth=1.5)

    # Adjust padding
    plt.subplots_adjust(left=0.025,
                        bottom=None,
                        right=None,
                        top=None,
                        wspace=None,
                        hspace=0.5)

    # Save figure
    plt.savefig(os.path.join(output_path, f'{patient_id}_{algo_name}.png'),
                dpi=300,
                bbox_inches="tight")
    plt.close()


def future_likelihood_plot(patient_id: str,
                           outputs: RiskOutput,
                           output_path: str,
                           add_score_line: bool = True) -> None:
    '''
    Generates a future likelihood plot
    '''

    likelihood_times_past = outputs.likelihood_times_past
    likelihoods_past = outputs.likelihoods_past
    likelihood_times = outputs.likelihood_times
    likelihoods = outputs.likelihoods
    med_cutoff = outputs.medium_thresholds_past[-1]
    high_cutoff = outputs.high_thresholds_past[-1]
    event_times = outputs.event_times

    # Convert UNIX timestamps to datetime
    convert_timestamps = lambda inp_list: [
        datetime.fromtimestamp(t / 1000) for t in inp_list
    ]
    # Future dates
    future_dates = convert_timestamps(likelihood_times)
    x_labels_future = np.arange(np.min(future_dates),
                                np.max(future_dates) + timedelta(hours=24),
                                timedelta(hours=24))
    # Past dates
    past_dates = convert_timestamps(likelihood_times_past)
    event_dates = convert_timestamps(event_times)

    # Find hightest and lowest likelihoods
    max_likelihood_idxs = np.where(np.array(likelihoods) == 1)[0]
    i = 0.95
    while not max_likelihood_idxs.any():
        max_likelihood_idxs = np.where(np.array(likelihoods) >= i)[0]
        i -= 0.05

    min_likelihood_idxs = np.where(np.array(likelihoods) == 0)[0]
    j = 0.05
    while not min_likelihood_idxs.any():
        min_likelihood_idxs = np.where(np.array(likelihoods) <= j)[0]
        j += 0.05

    # Plot
    _, (past, high, low) = plt.subplots(3, figsize=(22, 8))

    # Subplot - Past
    past.plot(event_dates, [1] * len(event_times),
              'red',
              linestyle='',
              marker='v',
              markersize=6)
    past.plot(past_dates, likelihoods_past, 'grey')

    # past plot labels / ticks
    past.xaxis.set_major_locator(dates.MonthLocator())
    past.xaxis.set_major_formatter(dates.DateFormatter('%b %y'))
    past.set_ylabel('Past Likelihood', fontsize=10, fontweight='bold')

    # Figure settings
    plt.setp((high, low),
             xlim=(np.min(future_dates), np.max(future_dates)),
             ylim=(0, 1.05))
    plt.setp(past,
             xlim=(np.min(past_dates), np.max(past_dates)),
             ylim=(0, 1.05))
    for ax in (past, low, high):
        ax.set_yticks((0, 1))
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)

    for ax in (low, high):
        # Subplot - High & Low Risk
        ax.plot(future_dates, likelihoods, 'k-', linewidth=1)
        ax.set_xticks(x_labels_future)
        ax.set_ylabel('Forecasted Likelihood', fontsize=10, fontweight='bold')

    # add future forecast lines (likelihoods and scores)
    low.plot(future_dates, [med_cutoff] * len(future_dates),
             color='green',
             linewidth=1.5)
    high.plot(future_dates, [high_cutoff] * len(future_dates),
              color='red',
              linewidth=1.5)
    if add_score_line:
        scores = get_risk_levels(
            np.array(likelihoods), med_cutoff,
            high_cutoff) / 4  # converts risk levels to 0.25, 0.5 and 0.75
        low.plot(future_dates, scores, color='grey', linewidth=0.5)
        high.plot(future_dates, scores, color='grey', linewidth=0.5)

    # Highlight high risk days
    max_ind = len(future_dates) - 1
    for max_likelihood_idx in max_likelihood_idxs:
        # Highlight high risk day
        high.axvspan(future_dates[max(0, max_likelihood_idx - 12)],
                     future_dates[min(max_ind, max_likelihood_idx + 12)],
                     color='red',
                     alpha=0.7)
        high.axvspan(future_dates[max(0, max_likelihood_idx - 12 * 4)],
                     future_dates[max(0, max_likelihood_idx - 12)],
                     color='red',
                     alpha=0.3)
        high.axvspan(future_dates[min(max_ind, max_likelihood_idx + 12)],
                     future_dates[min(max_ind, max_likelihood_idx + 12 * 4)],
                     color='red',
                     alpha=0.3)

    # Highlight low risk days
    for min_likelihood_idx in min_likelihood_idxs:
        low.axvspan(future_dates[max(0, min_likelihood_idx - 12)],
                    future_dates[min(max_ind, min_likelihood_idx + 12)],
                    color='green',
                    alpha=0.7)
        low.axvspan(future_dates[max(0, min_likelihood_idx - 12 * 4)],
                    future_dates[max(0, min_likelihood_idx - 12)],
                    color='green',
                    alpha=0.3)
        low.axvspan(future_dates[min(max_ind, min_likelihood_idx + 12)],
                    future_dates[min(max_ind, min_likelihood_idx + 12 * 4)],
                    color='green',
                    alpha=0.3)

    # Adjust padding
    plt.subplots_adjust(left=0.025,
                        bottom=None,
                        right=None,
                        top=None,
                        wspace=None,
                        hspace=0.5)

    # Save figure
    plt.savefig(os.path.join(output_path, f'{patient_id}.png'), dpi=300)
    plt.close()
