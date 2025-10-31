# Moving average

## Purpose

Simple algorithm using 90-day (default, but can be changed) moving average of seizure event rate to calculate likelihoods

## Required Inputs

patient_id
seizure_events.start_time

## Hyperparameters

- **moving_average_window_days**: int (default: 90) - number of days to use in moving average calculation
- **allow_shorter_windows_retrospective**: bool (default: False) - allows for shorter moving average windows if not enough events are found
- **allow_shorter_windows_prospective**: bool (default: True) - allows for shorter moving average windows for calculating future risk forecasts. If True, likelihoods *x* days away from the current day use a window duration of *moving_average_window_days - x* (i.e. it assumes no events have occurred between now and *x* days into the future). If False, likelihoods *x* days away from the current day use the same likelihood value as today.
- **threshold_increment**: float (default: 0.01) - increment between risk score thresholds to search
  over (eg default leads to threshold options: [0.01, 0.02, 0.03...0.99])
