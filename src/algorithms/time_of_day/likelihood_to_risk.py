from __future__ import annotations
from datetime import datetime
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray as Array
import pytz
from src.constants import MILLISECONDS_IN_AN_HOUR
from src.data import RiskOutput
from .utils import Parameters

# Threshold scoring weights
NONZERO_IN_LOW_WEIGHT = .01
UNREASONABLE_EVENTS_WEIGHT = .1
UNREASONABLE_TIMES_WEIGHT = .2

def round_daily(event_time: float):
    """
    Rounds the event time to the start of the day in UTC
    """
    datetime_time = datetime.fromtimestamp(event_time / 1000, pytz.timezone("UTC"))
    return datetime(datetime_time.year, datetime_time.month, datetime_time.day,
                        tzinfo=pytz.timezone("UTC")).timestamp() * 1000

def get_event_inds(likelihood_times: Array, event_times: Array, algo_type: str = "hourly") -> Array:
    """
    Finds indices of likelihood times where seizure events occur

    Parameters
    ----------
    likelihood_times: array of float
        time in UNIX timestamp
    event_times: array of float
        time in UNIX timestamp

    Returns
    -------
    event_indices: array of int
        indices of likelihood times where seizure events occur
    """
    if algo_type == "hourly":
        event_inds = np.unique([
            np.where(likelihood_times == event - event % MILLISECONDS_IN_AN_HOUR)[0][0]
            for event in event_times
            if event - event % MILLISECONDS_IN_AN_HOUR in likelihood_times
        ])
        return event_inds
    elif algo_type == "daily":
        event_inds = []
        for start_day in likelihood_times:
            end_day = start_day + MILLISECONDS_IN_AN_HOUR * 24
            event_inds.append(len([i for i in event_times if i>= start_day and i < end_day]))
        return np.where(event_inds)[0]



def get_risk_levels(likelihoods: Array, med_threshold: float, high_threshold: float) ->\
        Array:
    """
    Converts likelihoods to risk scores from provided risk thresholds

    Parameters
    ----------
    likelihoods: array of float
        forecast event likelihood [0-1]
    med_threshold: float
        threshold between low and medium risk
    high_threshold: float
        threshold between medium and high risk

    Returns
    -------
    scores: array of int

    """

    bounds = [0, med_threshold, high_threshold]
    scores = np.array([likelihoods >= bound for bound in bounds]).sum(axis=0)

    return scores


def get_thresholds_score(risk_levels: Array, event_indices: Array) ->\
        Tuple[float, List[float], List[float]]:
    """
    Determines the "score" for the given seizure events and risk levels.

    The higher the score, the better the risk levels. The score is calculated to favor:
    -more seizure events occuring in high risk
    -less time in high risk
    -less seizure events in low risk
    -more time in low risk

    Checks are also in place that reduce the score if the threhsolds result in an 'unreasonable'
    experience. For instance, the score is decreased if there are more seizure events in medium
    than in high risk. As the weights reduce the score, the lower the weight, the more important
    the requirement.

    Parameters
    ----------
    risk_levels: array of int
        risk levels from 1(low) to high(3)
    event_indices: array of int
        indices where seizure events occur

    Returns
    -------
    score: float
        representing how optimal the thresholds are
    events_in_risk_levels: list of float
        proportion of seizure events in each risk level
    time_in_risk_levels: list of float
        proportion of time in each risk level
    """

    if event_indices[-1] >= len(risk_levels):
        raise ValueError('Event indices out of range')

    event_indices = event_indices.astype(int)
    events_in_risk_levels = [
        np.sum(risk_levels[event_indices] == level) / len(event_indices) for level in range(1, 4)
    ]
    time_in_risk_levels = [np.sum(risk_levels == level) / len(risk_levels) for level in range(1, 4)]

    reasonable_times = time_in_risk_levels[0] < time_in_risk_levels[1] > time_in_risk_levels[2]
    reasonable_events = events_in_risk_levels[2] > events_in_risk_levels[1] >\
                        events_in_risk_levels[0]

    # Score maximises sz in high, time in low, and minimises time in high
    score = events_in_risk_levels[2] * (1 - time_in_risk_levels[2]) *\
            time_in_risk_levels[0] * (1 - events_in_risk_levels[0]) *\
            (1 if reasonable_times else UNREASONABLE_TIMES_WEIGHT) *\
            (1 if reasonable_events else UNREASONABLE_EVENTS_WEIGHT) *\
            (1 if not events_in_risk_levels[0] else NONZERO_IN_LOW_WEIGHT)

    return score, events_in_risk_levels, time_in_risk_levels


def get_thresholds(likelihoods: Array, event_inds: Array, inc: float) -> Tuple[float, float]:
    """
    Calculates the optimum thresholds for a single set of likelihoods and seizure events

    Parameters
    ----------
    likelihoods: array of float
        times in UNIX timestamps
    event_inds: array of int
    inc: float
        increment size when iterating through thresholds

    Returns
    -------
    best_med: float
    best_high: float
    """

    best_med = 0.25
    best_high = 0.75

    best_levels = get_risk_levels(likelihoods, best_med, best_high)
    best_score, _, _ = get_thresholds_score(best_levels, event_inds)

    for med_thresh in np.arange(inc, 1 - inc, inc):
        for high_thresh in np.arange(med_thresh + inc, 1, inc):
            risk_levels = get_risk_levels(likelihoods, med_thresh, high_thresh)
            score, _, _ = get_thresholds_score(risk_levels, event_inds)
            if score > best_score:
                best_score = score
                best_med = med_thresh
                best_high = high_thresh

    return best_med, best_high


def likelihood_to_risk(outputs: RiskOutput, params: Parameters) -> RiskOutput:
    """
    Converts likelihoods into risk levels, with optimised risk thresholds

    Parameters
    ----------
    outputs: RiskOutput
    params: Parameters

    Returns
    -------
    outputs: RiskOutput
    """

    likelihoods = np.array(outputs.likelihoods_past)
    likelihood_times = np.array(outputs.likelihood_times_past)
    event_times = np.array(outputs.event_times)

    events_inds = get_event_inds(likelihood_times, event_times)

    med_thresh, high_thresh = get_thresholds(likelihoods, events_inds, params.threshold_increment)
    med_thresholds = np.ones(likelihood_times.size) * med_thresh
    high_thresholds = np.ones(likelihood_times.size) * high_thresh

    outputs.medium_thresholds_past = med_thresholds.tolist()
    outputs.high_thresholds_past = high_thresholds.tolist()

    return outputs
