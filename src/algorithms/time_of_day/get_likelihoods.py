from typing import Tuple
import numpy as np
from numpy.typing import NDArray as Array
from src.constants import MILLISECONDS_IN_AN_HOUR, MILLISECONDS_IN_A_DAY, HOURS_IN_A_DAY
from src.data import RiskOutput, Number
from .utils import Parameters


def get_timestamps(event_times: Array, days_future: int,
                   request_time: Number) -> Tuple[Array, Array]:
    """
    Generate hourly timestamps for past and the required days into the future

    Parameters
    ----------
    event_times: array of float
        timestamps of seizure events, in days since epoch
    days_future: int
        number of days to forecast into the future
    request_time: Number

    Returns
    -------
    times_past: array of float
        past hourly times, in hours
    times_future: array of float
        future hourly times, in hours
    """
    # start of the hour immediately before the first event
    start_time = np.floor(event_times.min() * HOURS_IN_A_DAY)
    # hour after request was made
    end_time = np.ceil(request_time / MILLISECONDS_IN_AN_HOUR)

    times_past = np.arange(start_time, end_time)
    times_future = np.arange(end_time, end_time + 24 * days_future)

    return times_past, times_future


def get_likelihoods(event_times, likelihood_times_past, likelihood_times, params):
    """
    Calculate likelihoods for past events based on the time of day

    Parameters
    ----------
    event_times: array of float
        timestamps of seizure events, in days since epoch
    likelihoods_times_past: array of float
        past hourly times, in hours
    likelihood_times: array of float
        future hourly times, in hours
    params: Parameters

    Returns
    -------
    likelihoods_past: array of float
        past likelihoods, in range [0, 1]
    likelihoods: array of float
        future likelihoods, in range [0, 1]
    """
    # Initialise variables
    likelihoods_past = np.zeros_like(likelihood_times_past)
    likelihoods = np.zeros_like(likelihood_times)

    # convert event times to hours of the day
    events_tod = ((event_times % 1) * 24).astype(int)

    # Get likelhoods per hour of the day
    total_events = len(event_times)
    likelihoods_tod = {hour: np.sum(events_tod == hour) / total_events for hour in range(24)}

    # Assign likelihoods to past timestamps
    likelihood_times_past_tod = ((likelihood_times_past / 24 % 1) * 24).astype(int)
    likelihood_times_tod = ((likelihood_times / 24 % 1) * 24).astype(int)

    for i, hour in enumerate(likelihood_times_past_tod):
        likelihoods_past[i] = likelihoods_tod[hour]
    
    for i, hour in enumerate(likelihood_times_tod):
        likelihoods[i] = likelihoods_tod[hour]

    return likelihoods_past, likelihoods

def get_tod_likelihoods(outputs: RiskOutput, event_times: Array, params: Parameters,
                              request_time: Number) -> RiskOutput:
    """
    Parent function to get likelihood values from time of day

    Parameters
    ----------
    outputs: RiskOutput
    event_times: array of float
        time since epoch, in days
    params: Parameters
    request_time: Number

    Returns
    -------
    outputs: RiskOutput
    """

    likelihood_times_past, likelihood_times = get_timestamps(
        event_times, params.forecast_days, request_time)

    likelihoods_past, likelihoods = get_likelihoods(
        event_times, likelihood_times_past, likelihood_times, params)

    outputs.notes += f'Training events used for run: {len(event_times)}. Forecast generated for {params.forecast_days} days.'

    outputs.likelihoods = likelihoods.tolist(
    )  # list of float, 24*60 values for the next 60 days
    # list of float, likelihood values for past hours (0-1)
    outputs.likelihoods_past = likelihoods_past.tolist()
    # list of float,  hourly UNIX timestamps for 60 days
    outputs.likelihood_times = (likelihood_times *
                                MILLISECONDS_IN_AN_HOUR).tolist()
    # list of float, hourly UNIX timestamps for past data
    outputs.likelihood_times_past = (likelihood_times_past *
                                     MILLISECONDS_IN_AN_HOUR).tolist()
    # list of float, UNIX timestamps for past seizures
    outputs.event_times = np.array(event_times *
                                   MILLISECONDS_IN_A_DAY).tolist()

    return outputs
