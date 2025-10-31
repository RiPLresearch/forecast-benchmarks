import numpy as np

from src.algorithms.time_of_day.likelihood_to_risk import likelihood_to_risk
from src.algorithms.moving_average.calculate_moving_average import get_moving_average_likelihoods
from src.constants import MILLISECONDS_IN_A_DAY
from src.daily.daily_forecast import generate_daily_forecast
from src.data import RiskOutput, AlgorithmInputs, remove_non_lead_events
from .utils import Parameters


def run(inputs: AlgorithmInputs, outputs: RiskOutput,
        parameters: Parameters) -> RiskOutput:
    """
    Runs the risk forecast algorithm. Input and output structure are as required by the
    SEER-2250 spec.

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

    if not parameters.allow_shorter_windows_retrospective and (
        max(event_list) - min(event_list) < parameters.moving_average_window_days):
        # If not enough events to calculate moving average, return empty output
        outputs.save_forecasts = False
        if inputs.fail_early:
            return RiskOutput.build_empty(save_forecasts=False)

    outputs = get_moving_average_likelihoods(outputs, event_list, parameters, inputs.request_time)

    if not outputs.save_forecasts and inputs.fail_early:  # only triggers in run command
        return RiskOutput.build_empty(save_forecasts=False)

    outputs = likelihood_to_risk(outputs, parameters)

    if parameters.include_daily_forecast:
        outputs = generate_daily_forecast(outputs, parameters)
        parameters.remove_daily_forecast_padding()

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

