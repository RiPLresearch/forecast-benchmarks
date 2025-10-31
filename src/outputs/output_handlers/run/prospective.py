import os
from typing import List
import numpy as np
from src.constants import PATHS
from src.data import AlgorithmInputs, RiskOutput
from src.visual_utils import prospective_likelihood_plot
from src.utils import write_json


def output_handler(inputs: AlgorithmInputs, outputs: RiskOutput,
                   algo_name: str, randomized_inputs: List[AlgorithmInputs],
                   randomized_outputs: List[RiskOutput]) -> None:
    """Plots forecast"""
    save_path = PATHS.results_path()
    # save data
    outputs = add_non_lead_back(inputs, outputs)
    write_json(outputs.__dict__, os.path.join(save_path, f'{inputs.patient_id}_{algo_name}_results.json'))

    # plot forecast
    prospective_likelihood_plot(inputs.patient_id, outputs, save_path, algo_name)

    # save randomized data
    for i in range(len(randomized_inputs)):
        r_input = randomized_inputs[i]
        r_output = randomized_outputs[i]
        r_output = add_non_lead_back(r_input, r_output)
        write_json(
            r_output.__dict__, os.path.join(save_path, "randomized_outputs", f'{inputs.patient_id}_{algo_name}_random_{i}_results.json')
        )


def add_non_lead_back(inputs: AlgorithmInputs, outputs: RiskOutput):
    # add in other events (non lead)
    try:
        min_t, max_t = min(outputs.likelihood_times), max(
            outputs.likelihood_times)
    except ValueError:
        min_t, max_t = 0, 9e12

    event_list = np.array([
        event['start_time'] for event in inputs.seizure_events
        if min_t <= event['start_time'] <= max_t
    ])
    event_list.sort()
    outputs.event_times = event_list.tolist()
    return outputs
