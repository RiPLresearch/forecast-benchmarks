# Repo requirements
This repo uses Python 3.8. We recommend working in a new conda environment with Python 3.8 and the requirements in the requimrents.txt.
To get started:
1. `git clone https://github.com/RiPLresearch/forecast-benchmarks.git`
2. `pip install -r requirements.txt`

# How to store patient data and run the forecast

1. Save data in `cache` folder. Any seizure events should be stored under the 
2. Run forecast on patient data (described below in section 1 for pseudo-prospective or section 2 for retrospective/prospective outputs)
3. Validate forecast results for each patient (described in section 3, used only for pseudo-prospective testing)
4. If desired, build your own forecasting algorithm into the codebase (described in section 4)

## Format of saved data

See the following files for example input data:
`cache/seizure_events/eg1.json`

Saved as .json files in `cache/seizure_events` folder. Json file stores list of events like [{'start_time': 12343}, {'start_time': 45676}, {}, ...]
Note that all times are UNIX timestamps (in milliseconds)

# 1. How to run the forecast in pseudo-prospective mode

To run the forecast **pseudo-proecptively**, showing a hypothetical future forecast for all dates after training (currently default is the first 10 events):
`python -m src run --algos moving_average --ids all --outputs prospective -p`
The `-p` argument is what makes this code run psuedo-prospectively. If you don't add this flag, the algo will use all events to train and will produce a future forecast from time of running.

(Data is saved in `results`)

## Other arguments 

The `-n` argument allows you to select the minimum number of training seizure events (default is 10). This is useful for the pseudo-prospective forecast but also allows patients without enough data to be ignored in both the pseudo-prospective and prospective settings.

The `-mp` argument means the code will use multiprocessing. This may need to be removed, depending on your computer and processing requirements.

`--sig_test` runs randomly generated event times through algorithm 100 times to test significance of forecast algorithm. The `--sig_test` argument significantly slows down the code, and should be removed if the user is not interested in running significance tests.

## Algo types
There are two benchmark forecast algorithms that can be selected with the `--algos` argument:
- time_of_day: creates a forecast using time of day only
- moving_average: creates a forecast using a moving average window


## Specifying patient IDs 
The forecast will automaticaly run for all patients stored in the /cache folder, 
To select specific patients (e.g. P1 and P2), the follow can be run:
`python -m src run --algos overseer --ids P1,P2 --outputs prospective -p`
Note that the patient names should correspond to the name of the files in /cache (not inclusing the .json file extension)


## Forecast outputs
In pseudo-prospective mode, there is only one forecast output option that can be selected with the `--outputs` argument:
- prospective: stores training/testing (pseudo-prospective) forecast results alongside a snippet plot of the pseudo-prospective forecast

### Interpreting the psuedo-prospective output .json files
The `{patient_id}_{algo_name}_pseudoprospective_outputs.json` files are generated when the forecast is run is pseudo-prospective mode (i.e. using the `p` flag). The outputs are a dictionary with keys, explained below (all other keys are irrelevant):

- likelihoods: hourly pseudo-prospective forecasting likelihoods
- daily_likelihoods: daily pseudo-prospective forecasting likelihoods, starting at 12AM each day
- likelihood_times: hourly timestamps corresponding to the likelihoods
- daily_likelihood_times: daily timestamps corresponding to the daily_likelihoods
- event_times: seizure events that occured during the pseudo-prospective evaluation period
- medium_thresholds_past: medium thresholds (threshold between low and medium risk) corresponding to hourly likelihoods
- high_thresholds_past: high thresholds (threshold between medium and high risk) corresponding to hourly likelihoods
- medium_thresholds_daily: medium thresholds (threshold between low and medium risk) corresponding to daily likelihoods
- high_thresholds_daily: high thresholds (threshold between medium and high risk) corresponding to daily likelihoods

The `{patient_id}_{algo_name}_output_notes.json` files are also generate in this mode.
These notes are constructed every time the algorithm is trained. The pseudo-prospective output is desinged to retrain after each new seizure day, so timestamps (keys) correspond to days when new forecasts are generated and the notes are relevant to that specific run.

# 2. How to run the forecast in retrospective and prospective (real world use only) mode

To run the forecast in a real-world **prospective** setting, providing a future forecast from a specific date only (and noting that all events and outputs prior to that specific date are training):
`python -m src run --algos moving_average --ids all --outputs forecast,file`
The above command will also provide retrospective (training) likelihoods, which can be found in the output file by using the `file` option specified in `--outputs`:
`python -m src run --algos moving_average --ids all --outputs forecast,file`

Note that specifying algo names and patient IDs is done in the same way as described in section 1.

## Other arguments 

The `-n` argument allows you to select the minimum number of training seizure events (default is 10). This is useful for the pseudo-prospective forecast but also allows patients without enough data to be ignored in both the pseudo-prospective and prospective settings.

- likelihoods: hourly pseudo-prospective forecasting likelihoods
- daily_likelihoods: future daily likelihoods, starting from request_time ()
- likelihood_times: hourly timestamps corresponding to the likelihoods
- daily_likelihood_times: daily timestamps corresponding to the daily_likelihoods
- event_times: seizure events that occured during the pseudo-prospective evaluation period
- medium_thresholds_past: medium thresholds (threshold between low and medium risk) corresponding to hourly likelihoods
- high_thresholds_past: high thresholds (threshold between medium and high risk) corresponding to hourly likelihoods
- medium_thresholds_daily: medium thresholds (threshold between low and medium risk) corresponding to daily likelihoods
- high_thresholds_daily: high thresholds (threshold between medium and high risk) corresponding to daily likelihoods

## Forecast outputs
There are two forecast outputs in retrospective/prospective mode that can be selected with the `--outputs` argument:
- file: stores the forecast output in a json file
- forecast: plots a future forecast

### Interpreting the retrospective/propsective _prospective_outputs.json files
The `{patient_id}_{algo_name}_prospective_outputs.json` files are generated when the forecast is run with the `--outputs file` option.

#TODO

# 3. Validating the forecast results
After running the forecast in pseudo-prospective mdode, you can execute `python src/validation/validate_all.py` from your terminal.
This runs the `validate_all.py` script, which can be modified to include additional metrics if desired.
Note that this will ask for an input of the algorithm name.

Note that validation with can be run with or without significance testing. To add significance testing, run the initial forecast command with the `--sig_test` flag.

## Comparing results from two forecasts
If you would like to compare two forecasting algrorithms, you can execute `python src/validation/compare_algos.py` from your terminal.
This runs the `compare_algos.py` script, which can be modified to include additional metrics if desired.
This that this will ask for an input of the two algorithm names.


# 4. Building your own algorithm
To build and test your own algorithm in this environment, either for pseudo-prospective testing or for propsective forecast design, please copy the template folder `src/algorithms/template` to a separate folder, named as your algorithm, e.g. `src/algorithms/{my_algo_name_goes_here}`
The algorithm can be built into the main.py file in this new folder.
Run your algorithm using the command:
`python -m src run --algos {my_algo_name_goes_here} --ids all --outputs prospective -p`


# Contributing to this repository
If you would like to contribute to this repository by adding your own benchmarking algorithm or making changes to the existing codebase, please contact Rachel Stirling (rachel.stirling@unimelb.edu.au) or Pip Karoly (karolyp@unimelb.edu.au)


# Referencing this work
If you use this software, please cite it as below.
## TODO