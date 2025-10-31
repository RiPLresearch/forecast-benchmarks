import os
import click


@click.group()
def cli() -> None:
    pass


@cli.command(
    help=
    "For each patient ID, produces a single forecast per algorithm using historic data and uses" +\
    " the specified output handlers to process metrics and forecast results."
)
@click.option(
    "-i",
    "--ids",
    default="all",
    show_default=True,
    help=
    "Comma separated list of UUIDs indicating the patients to run the algorithms against."
)
@click.option(
    "-a",
    "--algos",
    default="moving_average",
    show_default=True,
    help="Comma separated list of algorithms to run. Defaults to 'risk_v1'.")
@click.option(
    "-o",
    "--outputs",
    default="forecast",
    help=
    "Comma separated list of output handlers that will run after the algorithms."
)
@click.option("-n",
              "--n_events",
              default="10",
              help="Number of training events to be used.")
@click.option("-p",
              "--prospective",
              is_flag=True,
              help="Set this flag to run the forecast prospectively")
@click.option(
    "-hyp",
    "--hyperparameters",
    default="default",
    show_default=True,
    help=
    "Comma separated list of json file names which contain the changes to be made "
    "to default hyperparameters. Set '_' or 'default' for any algorithm without "
    "changes")
@click.option(
    "-st",
    "--sig_test",
    is_flag=True,
    help="Set this flag to run the randomised outputs for significance testing"
)
@click.option("-mp",
              "--mulitprocessing",
              is_flag=True,
              help="Set this flag to run the commands with multiprocessing")
def run(algos, ids, outputs, n_events, prospective, hyperparameters, sig_test,
        mulitprocessing) -> None:

    os.environ["MIN_EVENTS"] = str(n_events)

    from src.runtime.algorithm import AlgorithmRuntime
    print("=== Running algorithm command ===")
    runtime = AlgorithmRuntime(
        ids.split(',') if ids else None,
        algos.split(',') if algos else ["overseer"],
        outputs.split(',') if outputs else ["print"], n_events, prospective,
        hyperparameters.split(',') if hyperparameters else ["default"],
        sig_test, mulitprocessing)
    print(
        f"Risk-algo Runtime initialised, starting {algos} for patients: {ids.split(',')}"
    )

    try:
        runtime.run()
    except MemoryError as mem_error:
        print(f"Task failed due to memory error: {mem_error}")
        raise mem_error
    except Exception as e:
        print(f"Task failed due to: {e}")
        raise e
