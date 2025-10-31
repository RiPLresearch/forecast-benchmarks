from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Any
from src.data import RequiredInputs, Number, AlgorithmParameters


@dataclass
class Parameters(AlgorithmParameters):
    """
    Extra algorithm parameters not universal to all algorithms. Universal parameters
    inherited from AlgorithmParameters. See the algorithm README for information on hyperparameters.
    """
    # pylint: disable=too-many-instance-attributes

    threshold_increment: float = 0.01

    def sanity_check(self) -> bool:
        '''
        Checks if provided parameters are out of expected range.
        Some parameters may not have limits if suitable limits are not known.
        '''
        # Some grouping to avoid too-many-branches
        raise_warning = False
        params = []

        # provide lower and upper (exclusive) limits for each variable in lists
        constrained_parameters: List[Tuple[str, Any, Number, Number]] = [
            ('threshold_increment', self.threshold_increment, 0., 0.1),
        ]
        for name, given, lower, upper in constrained_parameters:
            if isinstance(given, list):
                if not all(lower < i < upper for i in given):
                    raise_warning = True
                    params.append(name)
            else:
                if not lower < given < upper:
                    raise_warning = True
                    params.append(name)

        if raise_warning:
            print(
                f'Warning: {", ".join(params)} hyperparameter(s) outside the expected '
                'bounds')
        return not raise_warning and super().sanity_check()


def get_required_inputs() -> RequiredInputs:
    """
    Sets required inputs for time_of_day algo
    """
    required_inputs = RequiredInputs()
    required_inputs.seizure_events = True
    return required_inputs
