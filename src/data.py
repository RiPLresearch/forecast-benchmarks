# required for python 3.7-3.9 for returning Self from class
# e.g. see in RiskMetrics.build_empty() -> RiskMetrics
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union, Type, Dict, Any, List, Callable, Optional
from typing_extensions import TypedDict
import numpy as np
from src.constants import MILLISECONDS_IN_AN_HOUR

Number = Union[float, int]
NumberOrString = Union[str, float, int]

# Custom types. See https://github.com/python/typing/issues/182#issuecomment-870697089
JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any],
                  object]
JsonObject = Union[Dict[str, JsonValue], List[JsonValue]]



# pylint: disable=too-many-instance-attributes
# It doesn't make sense to split / merge any of these variables
@dataclass
class RiskOutput:
    """
    Output data from the risk algorithm
    """

    # list of float of hourly likelihoods
    likelihoods: Sequence[float] = field(default_factory=list)
    daily_likelihoods: Sequence[float] = field(default_factory=list)
    # list of float, likelihood values for past hours (0-1)
    likelihoods_past: Sequence[float] = field(default_factory=list)
    # list of float or int, hourly UNIX timestamps
    likelihood_times: Sequence[Union[int, float]] = field(default_factory=list)
    # list of float or int, daily UNIX timestamps
    daily_likelihood_times: Sequence[Union[int, float]] = field(
        default_factory=list)
    # list of float or int, hourly UNIX timestamps for past data
    likelihood_times_past: Sequence[Union[int,
                                          float]] = field(default_factory=list)
    # list of float or int, UNIX timestamps for past seizures
    event_times: Sequence[Union[int, float]] = field(default_factory=list)
    # list of float, thresholds between low and medium, for each likelihoods_past timestep
    medium_thresholds_past: Sequence[float] = field(default_factory=list)
    # list of float thresholds between medium and high, for each likelihoods_past timestep
    high_thresholds_past: Sequence[float] = field(default_factory=list)
    # list of float thresholds between low and medium, for each timezone; -12 to +14
    medium_thresholds_daily: Sequence[float] = field(default_factory=list)
    # list of float thresholds between medium and high, for each timezone; -12 to +14
    high_thresholds_daily: Sequence[float] = field(default_factory=list)
    # list of int, risk scores are 1 - Low, 2 - Medium and 3 - High risk
    risk_scores_future: Sequence[int] = field(default_factory=list)
    save_forecasts: bool = True
    notes: str = ''

    @staticmethod
    def build_empty(
            save_forecasts: bool = True) -> RiskOutput:
        return RiskOutput(save_forecasts=save_forecasts)

    def is_empty(self) -> bool:
        return self == self.build_empty() or self == self.build_empty() or self == self.build_empty(
                False) or self == self.build_empty(False)

    def check_structure(self):
        """
        Checks to make sure the length and contents of outputs makes sense. (not a quality check)
        """
        error_message = ''
        warning_message = ''
        expected_ranges = [
            (self.likelihoods, 'likelihoods', 0, 1),
            (self.likelihoods_past, 'likelihoods_past', 0, 1),
            (self.medium_thresholds_past, 'medium_thresholds', 0, 1),
            (self.high_thresholds_past, 'high_thresholds', 0, 1),
            (self.medium_thresholds_daily, 'medium_thresholds_daily', 0, 1),
            (self.high_thresholds_daily, 'high_thresholds_daily', 0, 1)
        ]

        if not len(self.likelihoods) == len(self.likelihood_times):
            error_message += 'Likelihoods and likelihood times do not have the same length. '
        if not len(self.likelihoods_past) == len(
                self.likelihood_times_past) == len(
                    self.medium_thresholds_past) == len(
                        self.high_thresholds_past):
            error_message += 'Mismatch between the lengths of past likelihoods, ' \
                                'likelihood times, and thresholds. '
        if not all(
                self.medium_thresholds_past[i] < self.high_thresholds_past[i]
                for i in range(len(self.medium_thresholds_past))):
            error_message += 'Invalid thresholds - medium higher than high for some values. '

        if not all(
                self.medium_thresholds_daily[i] < self.high_thresholds_daily[i]
                for i in range(len(self.medium_thresholds_daily))):
            error_message += 'Invalid daily thresholds - medium higher than high for some values. '

        for values, name, lower, upper in expected_ranges:
            if not all(lower <= value <= upper for value in values):
                error_message += f'{name} had values outside the required range of ({lower}, '\
                                 f'{upper}). '

        if error_message:
            raise ValueError(error_message)
        if warning_message:  # Not used at this point but adding for potential future use
            print(warning_message)
            return False
        return True


# We may wish to set frozen=True on this dataclass
@dataclass
class AlgorithmInputs:
    """
    Inputs for algorithms.

    Attributes
    ----------
    patient_id: str, the patient ID
    seizure_events: list, list of dicts,
        e.g. [{
               'start_time': 1622708683000,  # float, UNIX timestamps, ms
               'duration': 43986,  # float, milliseconds
               'awareness': 'Not Sure',  # str, options: Not sure/Aware/Unresponsive
               'motion': 'Movement',  # str, options: Not sure/Movement/None
               'size': 'Small',  # str, options: Small/Average/Big
               'typical' 'Yes'  # str, options: Yes/Not Sure
           }, {}, ...]
    """
    patient_id: str = ''

    seizure_events: Sequence[Mapping[str, NumberOrString]] = field(
        default_factory=list)

    non_input_fields = [
        'non_input_fields', 'patient_id', 'request_time', 'fail_early'
    ]
    request_time: Number = 0
    fail_early: bool = False

    def validate(self) -> None:
        """
        Checks the input fed into a risk forecast algorithm is correct.
        Raises ValueError if the inputs are not valid.
        """

        if not self.patient_id:
            raise ValueError('Patient ID is required')

        if not self.seizure_events:
            raise ValueError('Seizure event lists are empty')

        # removes duplications of events
        self.seizure_events = remove_duplicate_events(self.seizure_events)


@dataclass
class AlgorithmParameters:
    """
    Extra algorithm parameters for the forecast algorithm, universal to all algorithms.
    Default values must be provided.
    """

    def get_min_events() -> int:
        if os.environ.get("MIN_EVENTS"):
            return int(os.environ.get("MIN_EVENTS"))
        return 10 ## 10 is default

    forecast_days: int = 60
    generate_thresholds: bool = True
    min_events: int = get_min_events()
    include_daily_forecast: bool = True  # whether to include the daily likelihoods and thresholds
    lead_seizures: bool = False  # whether to use lead seizures only

    def apply_daily_forecast_padding(self: ParametersType, n_days: int = 1):
        """Adds padding to forecast days to calculate daily forecast"""
        if self.include_daily_forecast:
            self.forecast_days += n_days

    def remove_daily_forecast_padding(self: ParametersType, n_days: int = 1):
        """Removes padding to forecast days used to calculate daily forecast"""
        if self.include_daily_forecast:
            self.forecast_days -= n_days

    def sanity_check(self: ParametersType) -> bool:
        '''
        Raises a warning and returns False if hyperparameters are outside expected bounds.

        Code is structured to allow easy addition of more hyperparameters
        :return:
        '''
        raise_warning = False
        params = []
        if self.forecast_days <= 0:
            raise_warning = True
            params.append('forecast_days')
        if raise_warning:
            print(
                f'Warning: {", ".join(params)} hyperparameter(s) outside the expected bounds.'
            )
            return False
        return True



ParametersType = Union[AlgorithmParameters, Type[AlgorithmParameters]]


@dataclass
class RequiredInputs:
    """
    Inputs required for algorithm
    """
    seizure_events: bool = False

    # pylint: disable=no-member
    def all_true(self):
        for attribute in RequiredInputs.__dataclass_fields__:
            setattr(self, attribute, True)


class AlgoModule(TypedDict):
    name: str
    run: Callable[[AlgorithmInputs, RiskOutput, ParametersType], RiskOutput]
    check_inputs: Callable[[AlgorithmInputs, ParametersType, bool], bool]
    Parameters: Callable[[], ParametersType]
    get_required_inputs: Callable[[], RequiredInputs]


def remove_duplicate_events(
        events_list: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Only keeps last entry of event if event time is duplicated"""
    times = [event['start_time'] for event in events_list]
    last = [
        len(times) - 1 - times[::-1].index(time) for time in times
        if times.count(time) > 1
    ]
    events_list = [
        event for i, event in enumerate(events_list)
        if times.count(times[i]) == 1 or i in last
    ]
    return events_list


def remove_non_lead_events(seizure_events):
    # """Removes events that are not lead seizure events (defined as 3 or more seizures within 24h)"""
    """Keeping only the first seizure in a 4hr period"""
    if not seizure_events:
        return seizure_events

    seizure_events = sorted(seizure_events, key=lambda x: x['start_time'])
    event_list = np.array([event['start_time'] for event in seizure_events])

    sz_diff = np.diff(event_list)
    clusteres_ind = np.where(sz_diff <= MILLISECONDS_IN_AN_HOUR * 4)
    ind = (np.array(clusteres_ind) + 1).astype(int)
    # clusters = np.array(event_list)[ind]
    # clusters = [int(ts) for ts in clusters.flatten()]

    ind_sz = list(range(event_list.size))
    ind2 = np.delete(ind_sz, ind)
    leads = np.array(event_list)[ind2]
    leads = [int(ts) for ts in leads.flatten()]
    return [{"start_time": i} for i in leads]
