from datetime import datetime
import os
from typing import List
from src.constants import PATHS
from src.data import AlgorithmInputs, RiskOutput
from src.utils import write_json


def output_handler(inputs: AlgorithmInputs, outputs: RiskOutput,
                   algo_name: str, _randomized_inputs: List[AlgorithmInputs],
                   _randomized_outputs: List[RiskOutput]) -> None:
    """Saves metrics to a json file in src/results/algo_name"""

    save_fig = PATHS.results_path()

    # save data
    write_json(outputs.__dict__,
               os.path.join(save_fig, f'{inputs.patient_id}_{algo_name}_results.json'))
