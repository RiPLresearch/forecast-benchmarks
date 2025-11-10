from typing import List, Tuple
from datetime import datetime

import pytz
import numpy as np
from numpy.typing import NDArray as Array

from src.constants import MILLISECONDS_IN_A_DAY
from src.data import Number, ParametersType, RiskOutput


def choose_threshold_percentiles(likelihood_times: Array,
                                 event_times: Array) -> Tuple[Number, Number]:
    """
    Determine threshold percentiles based on past seizure frequency

    Parameters
    ----------
    likelihood_times: numpy array of float and int
        past likelihood times (in timestamp ms format)
    event_times: numpy array of float and int
        past event times (in timestamp ms format)

    Returns
    -------
    lower_perc: int or float
        lower percentile
    upper_perc: int or float
        higher percentile
    """

    # convert likelihood times and event times to days since epoch
    likelihood_days = np.unique(
        (likelihood_times / MILLISECONDS_IN_A_DAY).astype(int))
    event_days = np.unique((event_times / MILLISECONDS_IN_A_DAY).astype(int))

    # remove events outside of the likelihood times range
    event_days = event_days[event_days >= likelihood_days.min()]
    event_days = event_days[likelihood_days.max() >= event_days]

    # calculate past seizure frequency
    sz_freq = len(event_days) / len(likelihood_days)

    # choose lower and upper percentiles based on seizure frequency
    if sz_freq < 0.01:
        return (80, 95)
    elif sz_freq < 0.1:
        return (60, 90)
    elif sz_freq < 0.2:
        return (40, 75)
    elif sz_freq < 0.4:
        return (30, 60)
    return (25, 50)


def split_daily_likelihoods_by_timezones(
        likelihood_past: List[Number],
        likelihoods_future: List[Number]) -> List[List[Number]]:
    """
    Splits daily likelihoods represented each hour into daily likelihoods for each hour of the day

    Parameters
    ----------
    likelihood_past: list of float and int
        daily likelihoods represented for each hour in the past
    likelihood_future: list of float and int
        daily likelihoods represented for each hour in the future

    Returns
    -------
    timezone_lieklihoods: list of 24 lists, each of length = days in the past and future forecast
        each list within the list represents the likelihoods for a separate hour on the 24h clock
    """
    timezone_likelihoods = []
    for hour in range(24):
        hour_tz_past = likelihood_past[::-1][23 - hour::24]
        hour_tz_future = likelihoods_future[hour::24]
        timezone_likelihoods.append(hour_tz_past[::-1] + hour_tz_future)

    return timezone_likelihoods


def get_daily_thresholds(
        likelihoods_past: List[Number], likelihoods_future: List[Number],
        lower_perc: Number,
        upper_perc: Number) -> Tuple[List[Number], List[Number]]:
    """
    Convert likelihoods (past and future) to thresholds based on percentiles

    Parameters
    ----------
    likelihoods_past: list of float and int
        daily likelihoods represented for each hour in the past
    likelihoods_future: list of float and int
        daily likelihoods represented for each hour in the future
    lower_perc: float or int
        lower percentile
    upper_perc: float or int
        upper percentile

    Returns
    -------
    medium_thresholds: list (length 24)
        medium thresholds for each timezone
    high_thresholds: list (length 24)
        high thresholds for each timezone
    """

    if not 0 <= lower_perc < upper_perc <= 100:
        raise ValueError(
            f"Invalid percentile targets for daily forecast (0-100). Lower:"
            f" {lower_perc}, upper: {upper_perc}")

    timezone_likelihoods = split_daily_likelihoods_by_timezones(
        likelihoods_past, likelihoods_future)

    med_thresholds = [
        np.percentile(timezone_likelihoods[:][hr], lower_perc)
        for hr in range(24)
    ]
    high_thresholds = [
        max(np.percentile(timezone_likelihoods[:][hr], upper_perc),
            med_thresholds[hr] + .01) for hr in range(24)
    ]
    return med_thresholds, high_thresholds


def link_thresholds_to_timezones(likelihood_times, thresholds):
    """Converts 24 thresholds linked to the first 24 timesteps, into thresholds linked to timezone
     offset. Timezones range from -12 to +14 (27 values)"""

    first_time_utc_hour = datetime.fromtimestamp(likelihood_times[0] / 1000,
                                                 tz=pytz.utc).hour

    timezone_offsets = np.arange(-12, 15, 1)
    tz_thresholds = np.zeros(27)

    for offset in timezone_offsets:
        ind = -offset - first_time_utc_hour
        ind = ind % 24
        tz_thresholds[offset + 12] = thresholds[ind]

    return tz_thresholds


def generate_daily_forecast(outputs: RiskOutput,
                            _params: ParametersType, daily_likelihoods_complete: bool = False) -> RiskOutput:
    """
    Calculates daily forecast (likelihoods and thresholds)

    Parameters
    ----------
    outputs: RiskOutput
    _params: Parameters
    daily_likelihoods_complete: bool
        if True, daily likelihoods are already calculated and stored in outputs.daily_likelihoods

    Returns
    -------
    outputs: RiskOutput
        modified outputs
    """
    if daily_likelihoods_complete:
        # this is already calculated for moving average
        all_daily_likelihoods = outputs.daily_likelihoods[:-24]
    else:
        # get daily likelihoods
        all_hourly_likelihoods = np.array(
            list(outputs.likelihoods_past) + list(outputs.likelihoods))
        all_daily_likelihoods = [
            all_hourly_likelihoods[i:i + 24].sum() / 24
            for i in range(len(all_hourly_likelihoods) - 24)
        ]  # mean of the likelihoods # ignore last day

    # split daily likelihoods into past and future likelihoods
    n_past_hours = len(outputs.likelihoods_past)
    daily_likelihood_past = all_daily_likelihoods[:n_past_hours]
    daily_likelihoods_future = all_daily_likelihoods[n_past_hours:]
    outputs.daily_likelihoods = all_daily_likelihoods[n_past_hours -
                                                      24:]  # 24 hrs padding
    
    # Add times for daily likelihoods
    likelihood_times_past = list(outputs.likelihood_times_past)
    likelihood_times = list(outputs.likelihood_times)
    daily_likelihood_times = likelihood_times_past + likelihood_times
    outputs.daily_likelihood_times = daily_likelihood_times[n_past_hours - 24:-24]

    # choose threshold percentiles
    lower_perc, upper_perc = choose_threshold_percentiles(
        np.array(outputs.likelihood_times_past), np.array(outputs.event_times))

    # get daily thresholds for each hour in the 24h clock
    med_thresh, high_thresh = get_daily_thresholds(daily_likelihood_past,
                                                   daily_likelihoods_future,
                                                   lower_perc, upper_perc)
    # convert to 27 timeslots / timezones [-12 to +14]
    outputs.medium_thresholds_daily = link_thresholds_to_timezones(
        outputs.likelihood_times, med_thresh).tolist()
    outputs.high_thresholds_daily = link_thresholds_to_timezones(
        outputs.likelihood_times, high_thresh).tolist()

    # remove last day from forecasts
    outputs.likelihoods = outputs.likelihoods[:-24]
    outputs.likelihood_times = outputs.likelihood_times[:-24]
    outputs.risk_scores_future = outputs.risk_scores_future[:-24]

    return outputs
