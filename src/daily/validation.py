from datetime import datetime
from typing import Sequence, Tuple
import pytz

from src.daily.utils import (all_baseline_if_all_the_same,
                             calculate_likelihood_timestamps,
                             daily_high_if_all_hourly_high,
                             shift_likelihoods_for_timezone)
from src.data import Number, RiskOutput


def daily_for_validation(
    outputs: RiskOutput, timezone: int
) -> Tuple[Sequence[Number], Sequence[Number], float, float, Sequence[Number]]:
    """
    Processes daily forecast in RiskOutput in format used by validation.

    Parameters
    ----------
    outputs: RiskOutput
    timezone: int,
        timezone integer from UTC (default is Melbourne = +10)

    Returns
    -------
    future_daily_likelihoods: list of array of int/float
      Daily likelihood values for given timezone
    likelihood_dates: list of array of int/float
      Midnight timestamps (for given timezone) for each daily likelihood value
    medium_threshold: float
      Medium threshold for daily forecast
    high_threshold: float
      High threshold for daily forecast
    hourly_likelihoods: list of array of int/float
      Hourly likelihoods for each day in the future_daily_likelihoods sequence
      Length is 24 times the legnth of future_daily_likelihoods
    """

    forecast_days = int(len(outputs.likelihood_times) / 24)

    # Get daily thresholds
    medium_threshold = outputs.medium_thresholds_daily[timezone + 12]
    high_threshold = outputs.high_thresholds_daily[timezone + 12]

    # Find 'local' time of first future time (as index_offset)
    first_time_utc = datetime.fromtimestamp(outputs.likelihood_times[0] / 1000,
                                            tz=pytz.utc)
    index_offset = (first_time_utc.hour + timezone) % 24

    # Get daily and hourly likelihoods, adjusted for timezone, from the previous midnight
    future_daily_likelihoods, hourly_likelihoods = shift_likelihoods_for_timezone(
        outputs, forecast_days, index_offset)

    daily_likelihood_times = calculate_likelihood_timestamps(
        first_time_utc, timezone, forecast_days)

    # Enforce if all risk levels are high, daily risk is high
    high_threshold = daily_high_if_all_hourly_high(
        hourly_likelihoods, future_daily_likelihoods,
        [outputs.medium_thresholds_past[-1], outputs.high_thresholds_past[-1]],
        [medium_threshold, high_threshold])

    # if all daily risk levels are high / low, set all to medium
    medium_threshold, high_threshold = all_baseline_if_all_the_same(
        future_daily_likelihoods, medium_threshold, high_threshold)

    return (future_daily_likelihoods, daily_likelihood_times, medium_threshold,
            high_threshold, hourly_likelihoods)
