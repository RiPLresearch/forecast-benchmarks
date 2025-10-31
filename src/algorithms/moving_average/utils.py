from __future__ import annotations
from dataclasses import dataclass
from src.data import RequiredInputs, AlgorithmParameters


@dataclass
class Parameters(AlgorithmParameters):
    """
    Extra algorithm parameters not universal to all algorithms. Universal parameters
    inherited from AlgorithmParameters. See the algorithm README for information on hyperparameters.
    """
    moving_average_window_days: int = 90
    allow_shorter_windows_retrospective: bool = False # Allow shorter moving average windows if not enough events are present
    allow_shorter_windows_prospective: bool = True # Allow shorter moving average windows for calculating future risk forecasts
    threshold_increment: float = 0.01

    def sanity_check(self) -> bool:
        '''
        Checks if provided parameters are out of expected range.
        Some parameters may not have limits if suitable limits are not known.
        '''
        if self.moving_average_window_days < 1:
            raise ValueError(
                f"moving average days parameter 'moving_average_window_days' must be more than 1 day in duration, {self.moving_average_window_days} provided"
            )
        if self.moving_average_step_days < 1:
            raise ValueError(
                f"moving average step days parameter 'moving_average_step_days' must be more than 1 day in duration, {self.moving_average_step_days} provided"
            )


def get_required_inputs() -> RequiredInputs:
    """
    Sets required inputs for time_of_day algo
    """
    required_inputs = RequiredInputs()
    required_inputs.seizure_events = True
    return required_inputs
