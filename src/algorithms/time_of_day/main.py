import numpy as np
from src.algorithms.time_of_day.get_likelihoods import get_tod_likelihoods
from src.algorithms.time_of_day.likelihood_to_risk import likelihood_to_risk
from src.constants import MILLISECONDS_IN_A_DAY
from src.data import RiskOutput, AlgorithmInputs, remove_non_lead_events
from src.daily.daily_forecast import generate_daily_forecast
from .utils import Parameters


def run(inputs: AlgorithmInputs, outputs: RiskOutput,
        parameters: Parameters) -> RiskOutput:
    """
    Runs the risk forecast algorithm.

    Parameters
    ----------
    inputs: AlgorithmInputs
    outputs: RiskOutput
    parameters: Parameters

    Returns
    -------
    outputs: RiskOutput
    """

    try:
        inputs.validate()
    except ValueError:
        print("Invalid inputs for time_of_day algorithm")
        return RiskOutput.build_empty(save_forecasts=False)

    outputs.save_forecasts = True
    if not check_inputs(inputs, parameters):
        outputs.save_forecasts = False
        if inputs.fail_early:
            return RiskOutput.build_empty(save_forecasts=False)

    event_list = np.array(
        list(set(event['start_time'] for event in inputs.seizure_events))
    ) / MILLISECONDS_IN_A_DAY  # Event times in days
    event_list.sort()

    parameters.apply_daily_forecast_padding()

    outputs = get_tod_likelihoods(outputs, event_list, parameters,
                              inputs.request_time)
    outputs = likelihood_to_risk(outputs, parameters)

    if parameters.include_daily_forecast:
        outputs = generate_daily_forecast(outputs, parameters)
        parameters.remove_daily_forecast_padding()

    outputs.save_forecasts = check_outputs(
        outputs,
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


def check_outputs(outputs: RiskOutput,
                  fail_early=False) -> bool:
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
