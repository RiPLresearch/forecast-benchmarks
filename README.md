# Repo requirements
This repo uses Python 3.8. We recommend working in a new conda environment with Python 3.8 and the requirements in the requimrents.txt.
To get started:
1. `git clone https://github.com/RiPLresearch/forecast-benchmarks.git`
2. `pip install -r requirements.txt`

# To run and validate the forecast

1. Save data in `./cache` folder. Any seizure events should be stored under the 
2. Run forecast on patient data
3. Validate forecast results for each patient

## Format of saved data

See the following files for example input data:
`cache/seizure_events/eg1.json`

Saved as .json files in `./cache/seizure_events` folder. Json file stores list of events like [{'start_time': 12343}, {'start_time': 45676}, {}, ...]
Note that all times are UNIX timestamps (in milliseconds)

## How to run forecast

To run the forecast **pseudo-proecptively**, showing a hypothetical future forecast for all dates after training (currently default is the first 10 events):
`python -m src run --algos moving_average --ids all --outputs prospective -p`
The `-p` argument is what makes this code run psuedo-prospectively. If you don't add this flag, the algo will use all events to train and will produce a future forecast from time of running.

To run the forecast in a real-world **prospective** setting, giving future outputs from today only (and noting that all events and outputs prior to today are training):
`python -m src run --algos moving_average --ids all --outputs forecast`
The above command will also provide retrospective (training) likelihoods, which can be found in the output file by using the `file` option specified in `--outputs`:
`python -m src run --algos moving_average --ids all --outputs forecast,file`

(Data is saved in `./results`)

### Other arguments 

The `-n` argument allows you to select the minimum number of training seizure events (default is 10). This is useful for the pseudo-prospective forecast but also allows patients without enough data to be ignored in both the pseudo-prospective and prospective settings.

The `-mp` argument means the code will use multiprocessing. This may need to be removed, depending on your computer and processing requirements.

`--sig_test` runs randomly generated event times through algorithm 100 times to test significance of forecast algorithm. The `--sig_test` argument significantly slows down the code, and should be removed if the user is not interested in running significance tests.


### Selecting patients 

The forecast will automaticaly run for all patients stored in the /cache folder, 
To select specific patients (e.g. P1 and P2), the follow can be run:
`python -m src run --algos overseer --ids P1,P2 --outputs prospective -p`
Note that the patient names should correspond to the name of the files in /cache (not inclusing the file extension)


## Algo types
There are two benchmark forecast algorithms that can be selected with the `--algos` argument:
- time_of_day: creates a forecast using time of day only
- moving_average: creates a forecast using a moving average window


## Forecast outputs
There are three standard forecast outputs that can be selected with the `--outputs` argument:
- file: stores the forecast output in a json file
- forecast: plots a future forecast
- prospective: stores training/testing (pseudo-prospective) forecast results alongside a snippet plot of the pseudo-prospective forecast


## Validating the forecast results
Note that validation is designed to work for patients who have had significance testing run through the `--sig_test` flag, with the pseudo-prospective `-p` forecast type.

After running the forecast, you can execute `python src/validation/validate_all.py` from your terminal.
This runs the `validate_all.py` script, which can be modified to include additional metrics if desired.
Note that this will ask for an input of the algorithm name.


## Comparing results from two forecasts
If you would like to compare two forecasting algrorithms, you can execute `python src/validation/compare_algos.py` from your terminal.
This runs the `compare_algos.py` script, which can be modified to include additional metrics if desired.
This that this will ask for an input of the two algorithm names.


# Building your own algorithm
To build and test your own algorithm in this environment, either for pseudo-prospective testing or for propsective forecast design, please copy the template folder `src/algorithms/template` to a separate folder, named as your algorithm, e.g. `src/algorithms/{my_algo_name_goes_here}`
The algorithm can be built into the main.py file in this new folder.
Run your algorithm using the command:
`python -m src run --algos {my_algo_name_goes_here} --ids all --outputs prospective -p`


# Contributing to this repository
If you would like to contribute to this repository by adding your own benchmarking algorithm or making changes to the existing codebase, please contact Rachel Stirling (rachel.stirling@unimelb.edu.au) or Pip Karoly (karolyp@unimelb.edu.au)


# Referencing this work
If you use this software, please cite it as below.
## TODO