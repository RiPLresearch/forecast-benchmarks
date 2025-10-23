# Forecast Benchmarks
Basic algorithms used to benchmark event-based forecasting methods


## Exemplar data
Four example seizure diaries are provided (eg1 - eg4) as JSON files. Only the field   "start_time" is used by the code (where start time is the POSIX time in milliseconds).

## Moving average
Computes a basic 90-day moving average from daily seizure rates
