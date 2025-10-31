from typing import Tuple
import numpy as np
from numpy.typing import NDArray as Array
from src.data import Number
from src.constants import HOURS_IN_A_DAY, MILLISECONDS_IN_A_DAY, MILLISECONDS_IN_AN_HOUR



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


def get_likelihoods(event_times, likelihoods_times_past, likelihood_times, params):
    """
    Calculate likelihoods for past events based on the moving average window

    Parameters
    ----------
    event_times: array of float
        timestamps of seizure events, in days since epoch
    likelihoods_times_past: array of float
        past hourly times, in hours
    params: Parameters

    Returns
    -------
    likelihoods_past: array of float
        likelihoods for past events, in range [0, 1]
    """
    # Initialise variables
    likelihoods_past = np.zeros_like(likelihoods_times_past)
    likelihoods = np.zeros_like(likelihood_times)
    ma_window  = params.moving_average_window_days

    # remove duplicate events
    event_times = np.unique(event_times)

    # convert times to days
    likelihoods_times_past = likelihoods_times_past / HOURS_IN_A_DAY
    likelihood_times = likelihood_times / HOURS_IN_A_DAY

    for i, end_window in enumerate(likelihoods_times_past):

        if params.allow_shorter_windows_retrospective:
            start_window = max(end_window - ma_window, np.min(event_times))
        else:
            start_window = end_window - ma_window
        
        iter_window_length = end_window - start_window

        # Find number of events in moving average window
        events_in_window = np.where(
            (event_times >= start_window) & 
            (event_times < end_window)
        )[0]
        
        likelihoods_past[i] = len(events_in_window) / iter_window_length

    if params.allow_shorter_windows_prospective:

        for i, end_window in enumerate(likelihood_times):

            start_window =  max(end_window - ma_window, np.min(event_times))
            iter_window_length = end_window - start_window

            # Find number of events in moving average window
            events_in_window = np.where(
                (event_times >= start_window) & 
                (event_times < end_window)
            )[0]

            likelihoods[i] = len(events_in_window) / iter_window_length # Normalize to be in range [0, 1]

    else:
        likelihoods[:] = likelihoods_past[-1]  # Use last value for future likelihoods


    # Normalize likelihoods to be in range [0, 1]
    if any(likelihoods > 1):
        likelihoods = likelihoods / np.max(likelihoods)
    if any(likelihoods_past > 1):
        likelihoods_past = likelihoods_past / np.max(likelihoods_past)

    return likelihoods, likelihoods_past


def get_moving_average_likelihoods(outputs, event_times, params,
                              request_time):
    """
    Calculate moving average likelihoods based on seizure events
    """

    likelihood_times_past, likelihood_times = get_timestamps(
        event_times, params.forecast_days, request_time)
    
    first_event_day = np.floor(event_times.min()) * 24 # midnight of first event day, converted to hours
    ## if shorter windows allowed, first retrospective window can be min 1 day
    first_window_duration = 1 if params.allow_shorter_windows_retrospective else params.moving_average_window_days
    min_window_start = first_event_day + first_window_duration * 24 # convert to hours
    likelihood_times_past = likelihood_times_past[np.where(likelihood_times_past >= min_window_start)]

    if len(likelihood_times_past) == 0:
        outputs.save_forecasts = False
        return outputs

    likelihoods, likelihoods_past = get_likelihoods(event_times, likelihood_times_past, likelihood_times, params)

    outputs.likelihoods = likelihoods.tolist(
    )  # list of float, 24*60 values for the next 60 days
    # list of float, likelihood values for past hours (0-1)
    outputs.likelihoods_past = likelihoods_past.tolist()
    # list of float,  hourly UNIX timestamps for 60 days
    outputs.likelihood_times = (likelihood_times *
                                MILLISECONDS_IN_AN_HOUR).tolist()
    # list fo flo®at, hourly UNIX timestamps for past data
    outputs.likelihood_times_past = (likelihood_times_past *
                                     MILLISECONDS_IN_AN_HOUR).tolist()
    # list of float, UNIX timestamps for past seizures
    outputs.event_times = np.array(event_times *
                                   MILLISECONDS_IN_A_DAY).tolist()
    outputs.save_forcasts = True

    return outputs
