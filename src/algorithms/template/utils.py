from dataclasses import dataclass
from src.data import AlgorithmParameters, RequiredInputs, Number


@dataclass
class Parameters(AlgorithmParameters):
    """
    Extra algorithm parameters for the forecast algorithm
    """

    # pylint: disable=too-many-instance-attributes
    threshold_increment: float = 0.01

def get_required_inputs() -> RequiredInputs:
    """
    Sets required inputs for algo
    """
    required_inputs = RequiredInputs()
    required_inputs.seizure_events = True
    return required_inputs
