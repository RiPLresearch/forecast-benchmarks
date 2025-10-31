from datetime import date, datetime, timedelta
from typing import List, Sequence, Tuple
from dateutil import tz  # type: ignore
import numpy as np
from src.algorithms.time_of_day.likelihood_to_risk import get_risk_levels
from src.constants import MILLISECONDS_IN_A_DAY
from src.data import Number, RiskOutput


def daily_high_if_all_hourly_high(hourly_likelihoods: Sequence[Number],
                                  daily_likelihoods: Sequence[Number],
                                  hour_thresholds: List[Number],
                                  day_thresholds: List[Number]) -> Number:
    """
    If all hourly risk levels in a given day are high, ensure the daily risk level is high.
    Daily risk level is changed by changing the high threshold to be below all all-high days.

    Parameters
    ----------
    hourly_likelihoods: list of float,
        hourly likelihoods represented for each hour in the future
    daily_likelihoods: list of float,
        daily likelihoods represented for each hour in the future
    hour_thresholds: list of float
        [med_thresh, high_thresh]
    day_thresholds: list of float
        [med_thresh, high_thresh]

    Returns
    -------
    high_thresh: float,
        high daily threshold
    """

    # Get daily and hourly risk from likelihoods and thresholds
    daily_risk = get_risk_levels(np.array(daily_likelihoods), day_thresholds[0], day_thresholds[1])
    hourly_risk = get_risk_levels(np.array(hourly_likelihoods), hour_thresholds[0],
                                  hour_thresholds[1])
    hourly_risk = hourly_risk.reshape((int(hourly_risk.size / 24), 24))  # hourly grouped by days
    daily_sums = np.sum(hourly_risk, axis=1)

    high_thresh = day_thresholds[1]
    for i, daily_sum in enumerate(daily_sums):
        # high thresh is below the lowest daily_likelihood that should become high risk
        if daily_sum == 72 and daily_risk[i] < 3:
            high_thresh = min(daily_likelihoods[i] - .001, high_thresh)

    return high_thresh


def all_baseline_if_all_the_same(likelihoods: Sequence[Number], med_thresh: Number,
                                 high_thresh: Number) -> Tuple[Number, Number]:
    """
    If all future days have the same daily likelihood, ensure risk levels will all be medium.

    Parameters
    ----------
    likelihoods: list of float,
        daily likelihoods for every midnight timestamp
    med_thresh: float,
        medium daily threhsold
    high_thresh: float,
        high daily threshold

    Returns
    -------
    med_thresh: float,
        medium daily threhsold
    high_thresh: float,
        high daily threshold
    """
    if np.min(likelihoods) == np.max(likelihoods):
        med_thresh = med_thresh if med_thresh < likelihoods[0] else likelihoods[0] - 0.01
        high_thresh = high_thresh if high_thresh > likelihoods[0] else likelihoods[0] + 0.01

    return med_thresh, high_thresh


def shift_likelihoods_for_timezone(outputs: RiskOutput, forecast_days: int,
                                   index_offset: int) -> Tuple[Sequence[Number], Sequence[Number]]:
    """
    Shifts hourly and daily likelihoods so that the first likelihood is aligned with midnight.
    e.g. if the first likelihood is at 4am for the given timezone, shift 4 values backwards

    Parameters
    ----------
    outputs: RiskOutput
    forecast_days: int
        number of days to generate forecasts for
    index_offset: int
        hours to shift by, 0-23, i.e. hours since previous midnight

    Returns
    -------
    future_daily_likelihoods: list of float,
        one value for every day
    future_hourly_likelihoods: list of float,
        one value for every hour

    """

    # Get daily and hourly likelihoods, adjusted for timezone, from the previous midnight
    future_daily_likelihoods = outputs.daily_likelihoods[24 - index_offset::24]
    all_hourly_likelihoods = list(outputs.likelihoods_past) + list(outputs.likelihoods)
    hourly_likelihoods = all_hourly_likelihoods[-forecast_days * 24 - index_offset:]

    # trim
    future_daily_likelihoods = future_daily_likelihoods[:forecast_days]
    future_hourly_likelihoods = hourly_likelihoods[:forecast_days * 24]

    return future_daily_likelihoods, future_hourly_likelihoods


def calculate_likelihood_dates(first_time_utc: datetime, timezone: int,
                               forecast_days: int) -> List[date]:
    """
    Calculates the dates linked to the daily_likelihoods
    Parameters
    ----------
    first_time_utc: datetime.datetime,
        first timestamp in utc time
    timezone: int,
        from -12 to 14
    forecast_days: int

    Returns
    -------
    likelihood_dates: list of date
    """
    first_time_local = (first_time_utc + timedelta(hours=int(timezone))).replace(tzinfo=None)
    first_date = first_time_local.date()
    likelihood_dates = [first_date + timedelta(days=i) for i in range(forecast_days)]
    return likelihood_dates


def calculate_likelihood_timestamps(first_time_utc: datetime, timezone: int,
                                    forecast_days: int) -> List[Number]:
    """
    Calculates the timestamps linked to the daily_likelihoods
    Parameters
    ----------
    first_time_utc: datetime.datetime,
        first timestamp in utc time
    timezone: int,
        from -12 to 14
    forecast_days: int

    Returns
    -------
    likelihood_dates: list of int and float
    """
    tzlocal = tz.tzoffset('local', timezone * 3600)
    first_time_local = (first_time_utc + timedelta(hours=int(timezone))).replace(tzinfo=tzlocal)
    first_timestamp = first_time_local.replace(hour=0, minute=0, second=0,
                                               microsecond=0).timestamp() * 1000
    return np.arange(first_timestamp, first_timestamp + forecast_days * MILLISECONDS_IN_A_DAY,
                     MILLISECONDS_IN_A_DAY).tolist()
