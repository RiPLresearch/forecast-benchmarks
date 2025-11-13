import time
from typing import List, Mapping, Optional
import pandas as pd

from src.data import (AlgorithmInputs, RequiredInputs, NumberOrString, remove_duplicate_events)
from src.inputs.file_source import FileDataSource
from src.constants import MILLISECONDS_IN_A_SECOND


class DataGenerator:
    """
    Retrieves data from FileDataSource (stored data in cache/).
    """
    def __init__(self, source: FileDataSource):
        self.source = source

    def _get_seizure_events(
            self, patient_id: str, to_time: Optional[float] = None) -> List[Mapping[str, NumberOrString]]:
        """
        Opens seizure events, sorts and removes duplicates
        e.g.
           [{
            'start_time': 1622708683000,  # float, UNIX timestamps, ms
            }, {}, ...]
        """
        seizure_events = self.source.get_seizure_events(patient_id, to_time=to_time)
        print(f"[{patient_id}] Loaded seizure events from source")  # remove

        seizure_events.sort(key=lambda seizure_event: seizure_event['start_time'])

        return remove_duplicate_events(seizure_events)

    # pylint: disable=too-many-branches
    def generate_input(self, input_data: AlgorithmInputs, patient_id: str,
                       required_inputs: RequiredInputs, request_time: Optional[float] = None) -> AlgorithmInputs:
        """
        A wrapper function to generate the input.
        """

        # Log used inputs
        used_inputs = [
            key for key in required_inputs.__dict__
            if getattr(required_inputs, key)
        ]
        print(f"[{patient_id}] Generating inputs with {','.join(used_inputs)}")

        # New inputs class
        print(f"[{patient_id}] Generating inputs.")
        inputs = AlgorithmInputs(
            **{
                key: getattr(input_data, key)
                for key in required_inputs.__dict__
                if getattr(required_inputs, key)
            })
        inputs.patient_id = patient_id
        request_time = time.time() * MILLISECONDS_IN_A_SECOND if request_time is None else request_time
        inputs.request_time = request_time

        # Reset required inputs
        empty_input_class = AlgorithmInputs()
        required_inputs = RequiredInputs(
            **{
                key: True
                for key in required_inputs.__dict__
                if getattr(required_inputs, key) and (
                    getattr(inputs, key) == getattr(empty_input_class, key)
                    if not isinstance(getattr(inputs, key), pd.DataFrame) else
                    getattr(inputs, key).empty)
            })

        if required_inputs.seizure_events:
            inputs.seizure_events = self._get_seizure_events(patient_id, to_time=request_time)

        return inputs
