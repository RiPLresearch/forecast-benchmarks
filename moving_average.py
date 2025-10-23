#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:29:45 2025

@author: karolyp
"""

# This code computes a 90-day  moving average seizure rate and corresponding AUC score for next-day predictive performance

# run MA code prospectively
prospective = True

import json
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.metrics import roc_auc_score

# INSERT EG CODE HERE
with open("eg1.json", "r") as f:
    sz_data = json.load(f)


# EXTRACT SZ TIMES AS HORUS
sz_times = [sz['start_time'] for sz in sz_data]
sz_times_rel = [sz - min(sz_times) for sz in sz_times]
sz_hours = [sz / 1000 / 3600 for sz in sz_times_rel]


# CONVERT TO A DF WITH TIMEDELTA INDEX
td_objects = [timedelta(hours=sz) for sz in sz_hours]
td_index_objects = pd.TimedeltaIndex(td_objects)
df = pd.DataFrame(1, index=td_index_objects, columns=['Value'])

# Resample as a daily seizure rate and compute 90-day moving average
daily_rate = df.resample('1D').sum()
ma_daily = daily_rate.rolling(window=90).mean()

# Extract prospective (or retrospective) moving average for each day
if prospective:
    # start forecast from day 91, using previous day's moving average to forecast current day
    forecast = ma_daily[89:-1]['Value'].reset_index(drop=True)
else:
    # start forecast from day 91, using current moving average as the forecast (i.e. non causal)
    forecast = ma_daily[90:]['Value'].reset_index(drop=True)


# Normalise moving average to between 0 - 1 by min and max
min_val = np.min(forecast)
max_val = np.max(forecast)
forecast_prob = (forecast - min_val) / (max_val - min_val)

# convert daily seizure rate to a seizure index (True/False) beginning on day 91
sz_index = daily_rate[90:]['Value'].reset_index(drop=True)
sz_index = sz_index > 0


#Calculate the AUC-ROC score
auc_score = roc_auc_score(sz_index, forecast_prob)

print(f"The AUC-ROC score is: {auc_score}")
