import copy
from importlib import import_module
from multiprocessing.pool import Pool
import os
import time
from typing import Any, Callable, Tuple, cast, List, Optional

import numpy as np
from src.constants import MILLISECONDS_IN_A_DAY, PATHS
from src.daily.validation import daily_for_validation
from src.data import (AlgorithmInputs, Number, ParametersType, RiskOutput,
                      AlgoModule)
from src.inputs.data_generator import DataGenerator
from src.inputs.file_source import FileDataSource
from src.utils import get_algo, get_hyperparameters, get_training_inputs, round_daily, union_required_inputs, write_json

OutputHandler = Callable[[AlgorithmInputs, RiskOutput, str], None]
TIMEZONE: int = 0  # UTC timezone

N_SURROGATES = 100

cache_path = PATHS.cache_path()

class AlgorithmRuntime:
    # pylint: disable=too-many-instance-attributes
    ids: List[str]
    algos: List[AlgoModule]
    outputs: List[OutputHandler]

    def __init__(self, ids: List[str], algo_names: List[str],
                 outputs: List[str], n_events: str, prospective: bool,
                 hyperparams_change_files: List[str], sig_test: bool,
                 multiprocessing: bool, start_time: Optional[str]) -> None:
        """
        Creates a new algorithm runtime by reading in arguments from the namespace.
        Validates the arguments, including attempting to import the algorithm namespace.

        Parameters
        ----------
        args: args parsed by Namespace
        """
        self.ids = [
            i.replace(".json", "") for i in os.listdir(os.path.join(cache_path, "seizure_events"))
            if i.endswith(".json")
        ] if ids == ['all'] else ids
        self.algo_names = algo_names
        self.algos = list(
            filter(None, (get_algo(algo_name) for algo_name in algo_names)))
        self.outputs = list(
            filter(None, (AlgorithmRuntime.get_handler(handler_name)
                          for handler_name in outputs)))
        self.n_events = int(n_events)
        self.prospective = prospective
        self.data_generator = DataGenerator(FileDataSource())
        self.hyperparams_change_files = hyperparams_change_files
        self.sig_test = sig_test
        self.mp = multiprocessing

        # can only be used in retrospective/prospective mode
        self.request_time = None if start_time is None or self.prospective else float(start_time)

        # placeholders
        self.patient_inputs = None
        self.parameters = None
        self.algo_module = None

    @staticmethod
    def get_handler(handler_name: str) -> Optional[OutputHandler]:
        """
        Attempts to import an output handler by name (outputs are in the
        src/outputs/run folder)

        Parameters
        ----------
        handler_name: str, the handler folder to be imported

        Returns
        -------
        An output handler function if it matches the handler_name, otherwise None
        """
        try:
            handler = cast(
                Any,
                import_module(f'.outputs.output_handlers.run.{handler_name}',
                              'src'))
            if hasattr(handler, 'output_handler'):
                return cast(OutputHandler, handler.output_handler)

            print(
                f'{handler_name}.main does not contain an "output_handler" ' +
                'function, this output will not be generated')
            return None
        except ImportError:
            print(f'{handler_name} does not exist, ' +
                  'this output will not be generated')
            return None

    def get_hyperparameters_list(self) -> List[ParametersType]:
        """
        Generates hyperparameters classes for each algo based on hyperparameter change files.
        Default hyperparameters are used of no file is provided.
        """
        if self.hyperparams_change_files:
            if len(self.algos) == len(
                    self.hyperparams_change_files):  # one file listed per algo
                return [
                    get_hyperparameters(algo, self.hyperparams_change_files[i])
                    for i, algo in enumerate(self.algos)
                ]
            elif len(self.hyperparams_change_files
                     ) == 1:  # one file listed for all algos
                return [
                    get_hyperparameters(algo, self.hyperparams_change_files[0])
                    for algo in self.algos
                ]
            print(
                'Incorrect number of hyperparameter files provided, provide none, one,'
                ' or one for every algorithm')
            raise ValueError
        return [get_hyperparameters(algo, None) for algo in self.algos]

    def run(self) -> None:
        """
        Runs the requested algorithms. There are three stages to running algorithms:

        1. Generate inputs
        2. Run algorithms + generate metrics
        3. Generate outputs

        Inputs are standardised, and all the algorithms use the same input structure.
        """

        if not self.algos:
            print('Aborting algorithm runtime - no algorithms available')
            return

        if not self.ids:
            print('Aborting algorithm runtime - no IDs specified')
            return

        # Get parameters for each algorithm
        parameters = self.get_hyperparameters_list()
        print(f"Parameters: {parameters}")

        overall_start = time.perf_counter()
        for patient_id in self.ids:

            # check if result already exists
            # result_file_path = PATHS.results_path(
            #     f'{patient_id}_{self.algo_names[0]}_pseudoprospective_outputs.json')
            # if os.path.exists(result_file_path):
            #     print(
            #         f'[{patient_id}] Results already exist at {result_file_path}, skipping'
            #         ' algorithm run.')
            #     continue

            started = time.perf_counter()
            print(f"[{patient_id}] START generating forecasts (S000)")

            # Gather risk inputs
            risk_required_inputs = union_required_inputs(self.algo_names)

            try:
                risk_input = self.data_generator.generate_input(
                    AlgorithmInputs(), patient_id, risk_required_inputs, self.request_time)
            except Exception as e:
                print(
                    f'[{patient_id}] Failed to generate inputs due to: {e} (E000)')
                continue

            if not risk_input.seizure_events:
                print(
                    f'[{patient_id}] Skipping patient - no seizure data available '
                    '(E001)')
                continue

            risk_input.fail_early = True  # Fail early in prod
            inputs_done = time.perf_counter()
            print(
                f"[{patient_id}] Generated inputs in {inputs_done - started:.3f} sec "
                "(S001)")

            last = inputs_done
            generated_forecasts = False
            for i, algo in enumerate(self.algos):
                params = parameters[i]
                print(
                    f"[{patient_id}] Running {algo['name']} using parameters: {params}"
                )

                # run forecast (either prospectively using training events or from request time)
                try:
                    input_copy = copy.deepcopy(risk_input)
                    if self.prospective:
                        params.min_events = self.n_events
                        outputs = self._run_prospective_forecast(
                            input_copy, algo, params)
                    else:
                        outputs = self._run_algo(input_copy, algo, params)

                    if outputs.is_empty() or not outputs.likelihoods:
                        continue

                    if self.sig_test:
                        randomized_inputs = []
                        randomized_outputs = []
                        for _ in range(N_SURROGATES):
                            # run forecast (either prospectively using training events or from request time)
                            input_copy = copy.deepcopy(risk_input)

                            ## SHUFFLE EVENTS
                            events = np.array(
                                [e['start_time'] for e in input_copy.seizure_events])
                            random_events = np.random.uniform(low=events.min(),
                                                            high=events.max(),
                                                            size=events.size)
                            random_events.sort()

                            input_copy.seizure_events = [{
                                'start_time': e
                            } for e in random_events]

                            if self.prospective:
                                params.min_events = self.n_events
                                random_output = self._run_prospective_forecast(
                                    input_copy, algo, params)
                            else:
                                random_output = self._run_algo(
                                    input_copy, algo, params)

                            randomized_inputs.append(input_copy)
                            randomized_outputs.append(random_output)
                    else:
                        randomized_inputs = []
                        randomized_outputs = []

                    current = time.perf_counter()
                    print(
                        f"[{patient_id}] {algo['name']} ran in {current - last:.3f} sec"
                    )
                    last = current

                    if outputs.save_forecasts:
                        self._run_output_handlers(risk_input, outputs,
                                                algo['name'], randomized_inputs,
                                                randomized_outputs)
                        current = time.perf_counter()
                        print(
                            f"[{patient_id}] {algo['name']} handled outputs in {current - last:.3f}"
                            " sec")
                        last = current
                        generated_forecasts = True
                    else:
                        print(
                            f"[{patient_id}] {algo['name']} was not able to generate useful"
                            f" forecasts. Output handler not initiated.")
                
                except Exception as e:
                    print(
                        f"[{patient_id}] Failed to generate {algo['name']} forecasts due to: {e}"
                    )

            if generated_forecasts:
                print(f"[{patient_id}] FINISH Generated forecasts in "
                      f"{time.perf_counter() - started:.3f}")
            # except Exception as e:
            #     print(
            #         f"[{patient_id}] Failed to generate forecasts due to: {e}")

        print(
            f"Finished running algorithms in {time.perf_counter() - overall_start} secs"
        )

    def _run_algo(self, inputs: AlgorithmInputs, algo: AlgoModule,
                  parameters: ParametersType) -> RiskOutput:
        """Runs a single algorithm and returns the outputs + metrics"""
        outputs = RiskOutput.build_empty()
        return algo['run'](inputs, outputs, parameters)

    def _get_new_event_days(self, test_event_times):
        """Returns days to retrain the algorithm"""
        new_event_days = sorted(
            list(
                set(
                    round_daily(sz) + MILLISECONDS_IN_A_DAY
                    for sz in test_event_times)))

        return new_event_days

    def _run_algo_for_event_day(
        self, run_arguments: Tuple[Number, Number]
    ) -> RiskOutput:
        """Runs algo run module with given training end date and future days"""
        event_day, future_days = run_arguments

        # Forecast using training events
        inputs = get_training_inputs(self.patient_inputs, event_day)

        parameters_instance = copy.deepcopy(self.parameters)
        parameters_instance.forecast_days = int(future_days)
        return self.algo_module['run'](inputs, RiskOutput.build_empty(),
                                       parameters_instance)

    def train_test_split(self) -> Tuple[List[Number], List[Number]]:
        """
        Splits seizure events into training and testing events

        Returns
        -------
        training: list
            List of UNIX timestamps (ms)
        testing: list
            List of UNIX timestamps (ms)
        """
        # Sort events and ensure there are at least min_train_events
        events = sorted([
            float(event['start_time'])
            for event in self.patient_inputs.seizure_events
        ])
        if len(events) < self.n_events:
            return [], []

        test_from_idx = self.n_events
        # if events after min_train_events occur on same day, add them to the training list
        while test_from_idx < len(events) and round_daily(
                events[test_from_idx]) == round_daily(
                    events[test_from_idx - 1]):
            test_from_idx += 1
        return events[:test_from_idx], events[test_from_idx:]

    def _run_prospective_forecast(self, inputs: AlgorithmInputs,
                                  algo: AlgoModule,
                                  parameters: ParametersType) -> RiskOutput:
        """Runs single algorithm prospectively"""
        self.patient_inputs = inputs
        self.parameters = parameters
        self.algo_module = algo

        train, test = self.train_test_split()

        if not test:
            return RiskOutput()

        # Calculate days to retrain forecast and future likelihood times
        event_day = round_daily(max(train)) + MILLISECONDS_IN_A_DAY
        new_event_days = self._get_new_event_days(test)
        previous_event_days = np.array([event_day] + list(new_event_days)[:-1])
        future_days = (new_event_days -
                       previous_event_days) // MILLISECONDS_IN_A_DAY
        testing_days = list(zip(previous_event_days, future_days))

        to_remove = []
        for i, td in enumerate(testing_days):
            if td[1] == 0:
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            del testing_days[i]

        # run algo for each testing day
        if self.mp:
            with Pool(10) as p:
                days = [(i, j) for i, j in testing_days]
                outputs = p.map(self._run_algo_for_event_day, days)
        else:
            outputs = []
            days = [(i, j) for i, j in testing_days]
            for day in days:
                outputs.append(self._run_algo_for_event_day(day))

        # save notes
        output_notes = [{
            test_day[0]: output.notes
        } if not output.is_empty() else {test_day[0]: "No forecast generated"}
                        for output, test_day in zip(outputs, testing_days)]
        write_json(
            output_notes, os.path.join("results", f'{inputs.patient_id}_{algo["name"]}_output_notes.json'))

        # combine outputs
        combined_output = RiskOutput(event_times=test)

        for i, output in enumerate(outputs):
            # If no forecast provided, do not add to test outputs
            if output.is_empty():
                event_day, future_days = testing_days[i]
                continue

            combined_output.likelihoods += output.likelihoods
            combined_output.likelihood_times += output.likelihood_times
            combined_output.medium_thresholds_past += [
                output.medium_thresholds_past[-1]
            ] * len(output.likelihoods)
            combined_output.high_thresholds_past += [
                output.high_thresholds_past[-1]
            ] * len(output.likelihoods)

            # Append testing results of daily forecast
            (daily_likelihoods, daily_likelihood_times, med_thresh,
             high_thresh, _) = daily_for_validation(output, TIMEZONE)
            combined_output.daily_likelihoods += daily_likelihoods
            combined_output.daily_likelihood_times += daily_likelihood_times
            combined_output.medium_thresholds_daily += [
                med_thresh
            ] * len(daily_likelihoods)
            combined_output.high_thresholds_daily += [
                high_thresh
            ] * len(daily_likelihoods)

        return combined_output

    def _run_output_handlers(self, inputs: AlgorithmInputs,
                             outputs: RiskOutput, algo: str,
                             randomized_inputs: List[AlgorithmInputs],
                             randomized_outputs: List[RiskOutput]):
        """Runs output handlers for a given set of inputs/outputs and metrics"""
        for handler in self.outputs:
            handler(inputs, outputs, algo, randomized_inputs,
                    randomized_outputs)
