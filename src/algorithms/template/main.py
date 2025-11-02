from typing import Tuple
import numpy as np
from numpy.typing import NDArray as Array
from src.algorithms.time_of_day.likelihood_to_risk import likelihood_to_risk
from src.constants import MILLISECONDS_IN_AN_HOUR
from src.data import AlgorithmInputs, RiskOutput, remove_non_lead_events, Number
from src.daily.daily_forecast import generate_daily_forecast
from .utils import Parameters


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
    start_time = np.floor(event_times.min() / MILLISECONDS_IN_AN_HOUR)
    # hour after request was made
    end_time = np.ceil(request_time / MILLISECONDS_IN_AN_HOUR)

    times_past = np.arange(start_time, end_time) * MILLISECONDS_IN_AN_HOUR
    times_future = np.arange(end_time, end_time + 24 * days_future) * MILLISECONDS_IN_AN_HOUR

    return times_past, times_future


def run(inputs: AlgorithmInputs, outputs: RiskOutput,
        parameters: Parameters) -> RiskOutput:
    """
    Runs the risk forecast algorithm. All algorithms should be callable from a function
    of this name. Input and output structure are as required by the SEER-2250 spec.

    Parameters
    ----------
    inputs: AlgorithmInputs
    outputs: RiskOutput
    _parameters: Parameters

    Returns
    -------
    outputs: RiskOutput
    """

    try:
        inputs.validate()
    except ValueError:
        return RiskOutput.build_empty(save_forecasts=False)

    if not check_inputs(inputs, parameters):
        return RiskOutput.build_empty(save_forecasts=False)

    outputs.save_forecasts = True

    # ---------- ALGO CODE HERE ---------- #

    # run algorithm here and set values on output dataclass (values shown are for example purposes only)

    ## EVENT TIMES USED ##
    event_list = np.array(
        list(set(event['start_time'] for event in inputs.seizure_events))
    )  # Event times in days
    event_list.sort()

    # This is usually toggled on, so daily forecasts are generated as 24h averages of hourly forecasts
    parameters.apply_daily_forecast_padding()

    ## NOW ADD LIKELIHOODS TO OUTPUTS ##
    # Variables ending in _past are past predictions (retrospective)
    # Variables without are future predictions (prospective)
    # Note that likelihood_times_past and likelihood_times must be hourly timestamps, in milliseconds since epoch

    likelihood_times_past, likelihood_times = get_timestamps(event_list, parameters.forecast_days, inputs.request_time)

    # These are past predictions (retrospective)
    outputs.likelihood_times_past = likelihood_times_past.tolist()

    # INSERT ALGORITHM HERE. Using random number generator between 0 and 1 as example here
    outputs.likelihoods_past = np.random.rand(len(likelihood_times_past)).tolist()

    # These are future predictions (prospective)
    outputs.likelihood_times = likelihood_times.tolist()

    # INSERT ALGORITHM HERE. Using random number generator between 0 and 1 as example here
    outputs.likelihoods = np.random.rand(len(likelihood_times)).tolist()

    # Add any notes about the forecast run here
    outputs.notes = f'Training events used for run: {len(event_list)}. Forecast generated for {parameters.forecast_days} days.'

    outputs.event_times = event_list.tolist()

    # Convert likelihoods to risk levels
    # This code uses a basic optimization algorithm to choose medium and high risk thresholds
    outputs = likelihood_to_risk(outputs, parameters)

    if parameters.include_daily_forecast:
        outputs = generate_daily_forecast(outputs, parameters)
        parameters.remove_daily_forecast_padding()
    
    # ---------- END ALGO CODE ---------- #

    outputs.save_forecasts = check_outputs(outputs,
                                           fail_early=inputs.fail_early)

    return outputs


def check_inputs(inputs: AlgorithmInputs,
                 parameters: Parameters,
                 _modify_inputs: bool = False) -> bool:
    """
    Checks whether input is usable
    """
    if parameters.lead_seizures:
        inputs.seizure_events = remove_non_lead_events(inputs.seizure_events)

    if len(inputs.seizure_events) < parameters.min_events:
        return False
    return True


def check_outputs(outputs: RiskOutput, fail_early=False) -> bool:
    '''
    All final checks for whether forecasts should be used
    '''
    if not outputs.check_structure():
        return False
    if not outputs.save_forecasts:
        return False
    if fail_early:
        # Room to add performance requirements here if needed (maybe only run if fail_early is true)
        pass

    return True
