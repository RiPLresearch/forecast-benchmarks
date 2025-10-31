from typing import List
from src.constants import PATHS
from src.data import AlgorithmInputs, RiskOutput
from src.visual_utils import future_likelihood_plot

ADD_SCORE_LINE = False


def output_handler(inputs: AlgorithmInputs, outputs: RiskOutput,
                   _algo_name: str, _randomized_inputs: List[AlgorithmInputs],
                   _randomized_outputs: List[RiskOutput]) -> None:
    """Plots forecast"""
    save_fig = PATHS.results_path()
    future_likelihood_plot(inputs.patient_id,
                           outputs,
                           save_fig,
                           add_score_line=ADD_SCORE_LINE)
