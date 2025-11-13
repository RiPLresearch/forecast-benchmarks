import copy
import os
import json
import pickle
import datetime
import shutil
from typing import Any, List, Optional, Mapping, Sequence, cast
from importlib import import_module
import numpy as np
import pandas as pd
import pytz
from src.constants import DAYS_IN_A_YEAR, MILLISECONDS_IN_A_SECOND
from src.data import ParametersType, RequiredInputs, AlgoModule, AlgorithmInputs, RiskOutput, Number
from src.constants import MILLISECONDS_IN_A_DAY, HOURS_IN_A_DAY, MILLISECONDS_IN_AN_HOUR, PATHS


def get_hyperparameters(algo: AlgoModule,
                        hyperparameters_file: Optional[str]) -> ParametersType:
    """
    Updates the parameters based upon the specified json file. Returns default if no file given.

    Parameters
    ----------
    algo: dict
        algorithm module, as returned by get_algo
    hyperparameters_file: str, optional,
        json file (without .json) where changes are stored.
        File should be within algorithms/<algo_name>/hyperparameters.

    Returns
    -------
    updated: <algo_name>.utils.Parameters
        Parameters class, specific to the algorithm, with required updates
    """

    default_strings = ['_', 'default']
    default = algo['Parameters']

    if not hyperparameters_file or hyperparameters_file in default_strings:
        return default()

    # Load specified json
    try:
        h = read_json(
            PATHS.algo_path(algo['name'], 'hyperparameters',
                            f'{hyperparameters_file}.json'))
    except FileNotFoundError as e:
        print(
            f"Could not find {hyperparameters_file}.json for the {algo['name']} "
            "algorithm")
        raise e

    # update parameters
    updated = default()
    for key in h:
        updated.__dict__[key] = h[key]

    return updated


def get_algo(algo_name: str) -> Optional[AlgoModule]:
    """
    Attempts to import an algorithm by name (algorithms are in the src/algorithms folder).

    Parameters
    ----------
    algo_name: str, the algorithm folder to be imported

    Returns
    -------
    The algorithm "run" function if importable, None otherwise
    """

    if algo_name == 'template':
        print('Ignoring "template" algorithm')
        return None

    try:
        algo = cast(Any, import_module(f'.algorithms.{algo_name}.main', 'src'))
        utils = cast(Any, import_module(f'.algorithms.{algo_name}.utils',
                                        'src'))

        if hasattr(algo, 'run') and hasattr(utils, 'Parameters') and hasattr(
                utils, 'get_required_inputs'):
            return {
                "name": algo_name,
                "run": algo.run,
                "check_inputs": algo.check_inputs,
                "Parameters": utils.Parameters,
                "get_required_inputs": utils.get_required_inputs
            }

        print(f'{algo_name}.main does not contain a "run" ' +\
                     'function, or utils does not contain Parameters class or get_required_inputs '
                     'function. This algorithm will not be run')
        return None
    except ImportError:
        print(f'{algo_name}.main or .utils does not exist. '
              'This algorithm will not be run')
        return None


def check_algo(algo_name: str) -> Optional[str]:
    """
    Attempts to import an algorithm by name (algorithms are in the src/algorithms folder).

    Parameters
    ----------
    algo_name: str, the algorithm folder to be imported

    Returns
    -------
    algo_name if algorithm exists, None otherwise
    """
    return None if get_algo(algo_name) is None else algo_name


def read_json(file_path: str) -> Any:
    """De-serializes file to Python object."""
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(data: Any, file_path: str) -> Any:
    """Serializes and writes Python object to file."""
    # ensure dir exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=True)
    return data


def read_pickle(file_path: str) -> Any:
    """Reads pickle file to Python object."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle(data: Any, file_path: str) -> Any:
    """Writes Python object to pickle file and returns object."""
    # ensure dir exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def algo_required_inputs(algo_name: str) -> RequiredInputs:
    """Returns RequiredInputs class of required inputs from algo name"""
    algo_module = cast(Mapping[str, Any], get_algo(algo_name))
    return algo_module['get_required_inputs']()


def union_required_inputs(algo_names: List[str]) -> RequiredInputs:
    """
    Get the union of all required inputs for each algo

    Parameters
    ----------
    algo_names: list of str
        List of algo names

    Returns
    -------
    RequiredInputs class containing all input options
        Attributes of class (input options) are set to True if any of the
        algos in algo_modules contain a True value for that attribute
    """
    union_of_inputs = RequiredInputs().__dict__
    for algo in algo_names:
        algo_inputs = algo_required_inputs(algo).__dict__
        for key in algo_inputs:
            union_of_inputs[key] = union_of_inputs[key] or algo_inputs[key]
    return RequiredInputs(**union_of_inputs)


def timestamp_to_datetime_str(timestamp: float,
                              string_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Converts time in milliseconds to a datetime string required by ql. Fractions of seconds are
    rounded down. Does not account for timezones.

    Parameters
    ----------
    timestamp: float
        UTC timestamp in milliseconds
    string_format: str
        string format of date (default = "%Y-%m-%d %H:%M:%S")

    Returns
    -------
    dt: str
        format: 'YYYY-MM-DD HH:MM:SS'
    """

    dt = datetime.datetime.utcfromtimestamp(timestamp /
                                            MILLISECONDS_IN_A_SECOND)
    return dt.strftime(string_format)


def delete_path(path: str) -> None:
    """Delete folder or file path"""
    try:
        shutil.rmtree(path, ignore_errors=False)
    except (NotADirectoryError, FileNotFoundError):
        os.remove(path)


def random_forecasts(inputs: AlgorithmInputs, params: ParametersType, outputs: RiskOutput) -> \
        RiskOutput:
    """
    Generates RiskOutput with randomly generated forecasts, useful when validating.
    """

    # Get event times
    event_times = np.array(
        list(set(event['start_time'] for event in inputs.seizure_events))
    ) / MILLISECONDS_IN_A_DAY  # Event times in days
    event_times.sort()
    outputs.event_times = (event_times * MILLISECONDS_IN_A_DAY).tolist()

    # Get timestamps
    # start of the hour immediately before the first event
    start_time = np.floor(event_times.min() * HOURS_IN_A_DAY) if len(event_times) else \
        np.floor(inputs.request_time / MILLISECONDS_IN_AN_HOUR)
    # hour after request was made
    end_time = np.ceil(inputs.request_time / MILLISECONDS_IN_AN_HOUR)

    outputs.likelihood_times_past = (np.arange(start_time, end_time) *
                                     MILLISECONDS_IN_AN_HOUR).tolist()
    outputs.likelihood_times = (
        np.arange(end_time, end_time + 24 * params.forecast_days) *
        MILLISECONDS_IN_AN_HOUR).tolist()

    past_steps = len(outputs.likelihood_times_past)
    future_steps = len(outputs.likelihood_times)

    # Random forecasts
    rng = np.random.default_rng(2021)
    likelihoods = rng.random(future_steps)
    likelihoods_past = rng.random(past_steps)

    likelihoods[likelihoods == 0] = 1e-9
    likelihoods[likelihoods == 1] = 1 - 1e-9
    likelihoods_past[likelihoods_past == 0] = 1e-9
    likelihoods_past[likelihoods_past == 1] = 1 - 1e-9

    outputs.likelihoods = likelihoods.tolist()
    outputs.likelihoods_past = likelihoods_past.tolist()

    # assume thresholds
    outputs.medium_thresholds_past = (np.ones(past_steps) * 0.25).tolist()
    outputs.high_thresholds_past = (np.ones(past_steps) * 0.75).tolist()

    outputs.notes = '|| Random forecast ||'
    outputs.save_forecasts = False
    return outputs


def merge_close_lists(inputs: Sequence[Sequence[Number]],
                      decimal_places: int = 1):
    output = []
    for seq in inputs:
        rounded = np.round(seq, decimal_places)
        output.extend(rounded)

    return np.sort(np.unique(output)).tolist()


def round_hourly(date: Number) -> Number:
    '''
    Rounds UNIX timestamps down to most recent hour
    (e.g. Jan 1 2020 at 12:45PM rounds to Jan 1 2020 at 12PM)

    Parameters
    ----------
    date: float (UNIX timestamps in milliseconds)
        Date to round

    Returns
    -------
    rounded date: float (UNIX timestamps in milliseconds)
        Rounded date
    '''
    if isinstance(date, (int, float, np.int64)):
        datetime_time = datetime.datetime.fromtimestamp(
            date / 1000, pytz.timezone("UTC"))
        return datetime.datetime(datetime_time.year, datetime_time.month,\
                        datetime_time.day, datetime_time.hour,
                        tzinfo=pytz.timezone("UTC")).timestamp() * 1000
    raise TypeError('Needs float or int format. Currently {}.'.format(
        type(date)))


def round_daily(date):
    '''
    Rounds UNIX timestamps down to the last day (last 12AM occurence)
    (e.g. Jan 1 2020 at 8:00PM rounds to Jan 1 2020 at 12AM)

    Parameters
    ----------
    date: float (UNIX timestamps in milliseconds)
        Date to round

    Returns
    -------
    rounded date: float (UNIX timestamps in milliseconds)
        Rounded date
    '''
    if isinstance(date, (int, float, np.int64)):
        datetime_time = datetime.datetime.fromtimestamp(
            date / 1000, pytz.timezone("UTC"))
        return datetime.datetime(
            datetime_time.year,
            datetime_time.month,
            datetime_time.day,
            tzinfo=pytz.timezone("UTC")).timestamp() * 1000
    else:
        raise TypeError('Needs float or int format. Currently {}.'.format(
            type(date)))


def filter_between_times_dataframe(
        df: pd.core.frame.DataFrame,
        from_time: Number = 0,
        to_time: Number = 9e12,
        column: str = 'timestamp') -> pd.core.frame.DataFrame:
    '''
    Filters out timestamps of dataframe outside of from_time and to_time

    Parameters
    ----------
    df: dataframe with {column} column (default = 'timestamp')
    from_time: UNIX timestamp (ms) (default = 0)
    to_time: UNIX timestamp (ms) (default = 9e12)

    Returns
    ----------
    df: filtered dataframe
    '''
    try:
        return df[(df[column] >= from_time)
                  & (df[column] <= to_time)].reset_index(drop=True)
    except KeyError as e:
        print(f'Dataframe must contain {column} column.')
        raise e


def get_training_inputs(inputs: AlgorithmInputs,
                        end_train: Number) -> AlgorithmInputs:
    """
    Removes inputs after the end_train date

    Parameters
    -------
    inputs: AlgorithmInputs
        Inputs for algo for single patient
    end_train: UNIX timestamp (ms)

    Returns
    -------
    AlgorithmInputs
        Filtered inputs
    """
    predicate = lambda data, field: cast(Number, data[field]) < end_train

    new_inputs = copy.deepcopy(inputs)
    new_inputs.seizure_events = [
        event for event in new_inputs.seizure_events
        if predicate(event, 'start_time')
    ]
    new_inputs.request_time = end_train
    return new_inputs
