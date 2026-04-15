import copy
from datetime import datetime
import os
import click
import yaml


from vista.evaluate import evaluate
from vista.utils.logger import get_logger
from vista.utils.utils import load_yaml
from vista.utils.grid import create_experiment
from vista.utils.run import ParallelRun


OUT_FOLDER = "out"

FUNCTIONS = {
    "evaluate": evaluate,
}
FUNCTION_SLURMS = {
    "evaluate": "hpc/launch_run",
    "computational": "hpc/launch_computational",
}
FUNCTIONS_CONDOR = {
    "evaluate": "hpc/condor",
    "computational": "hpc/condor",
}
SCHEDULERS = {
    "slurm": FUNCTION_SLURMS,
    "condor": FUNCTIONS_CONDOR,
}


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


def manage_multiprocess_run(run_parameters, run_name, logger, job_parallelism=None):
    """
    Manage the multiprocess run.
    This function is used to launch the run in parallel or sequentially.
    """
    if "dataloader" in run_parameters and "num_processes" in run_parameters["dataloader"]:
        multi_runs = [
                copy.deepcopy(run_parameters) for _ in range(run_parameters["dataloader"]["num_processes"])
            ]
        if job_parallelism is not None and job_parallelism > 1:
            run_names = [
                f"{run_name}/job_{str(i//job_parallelism).zfill(3)}/p_{str(i%job_parallelism).zfill(3)}" for i in range(run_parameters["dataloader"]["num_processes"])
            ]
        else:    
            run_names = [
                f"{run_name}/p_{str(i).zfill(3)}" for i in range(run_parameters["dataloader"]["num_processes"])
            ]
        
        logger.info(f"Running {len(multi_runs)} processes in parallel")
        for i, run_parameters in enumerate(multi_runs):
            run_name = f"{run_names[i]}"
            multi_runs[i]["dataloader"]["process_id"] = i
            os.makedirs(run_name, exist_ok=True)
    else:
        multi_runs = [run_parameters]
        run_names = [run_name]
        logger.info("Running in single process mode")

    return multi_runs, run_names


@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run the grid in parallel",
)
@click.option(
    "--only_create",
    default=False,
    is_flag=True,
    help="Only create the slurm scripts",
)
@click.option(
    "--function",
    default="evaluate",
    help="Name of the function to run, either 'evaluate' only for now",
)
@click.option(
    "--job_parallelism",
    default=None,
    type=int,
    help="If --parallel is provided, and dataloader.num_processes is provided in the parameters, this will set the number of processes to use in a single job, should be divisor of dataloader.num_processes",
)
@click.option(
    "--scheduler",
    default="slurm",
    help="Scheduler to use, either 'slurm' or 'condor'",
)
def grid(parameters, parallel, only_create=False, function="evaluate", job_parallelism=None, scheduler="slurm"):
    assert function in FUNCTIONS, f"Function {function} not recognized, available functions: {list(FUNCTIONS.keys())}"
    
    run_function = FUNCTIONS[function]
    slurm_script = SCHEDULERS[scheduler][function]
    
    assert os.path.exists(slurm_script), f"Slurm script {slurm_script} does not exist"
    
    parameters = load_yaml(parameters)
    grid_name = parameters["grid"]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{grid_name}"
    log_folder = os.path.join(OUT_FOLDER, grid_name)
    
    runs_parameters = create_experiment(parameters)
    
    os.makedirs(log_folder)
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)

    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        run_name = f"{log_folder}/run_{i}"
        os.makedirs(run_name, exist_ok=True)
        multi_runs, run_names = manage_multiprocess_run(
            run_parameters, run_name, grid_logger, job_parallelism
        )
        if parallel:
            job_parallelism = job_parallelism if job_parallelism is not None else 1
            jobs = len(multi_runs) // job_parallelism
            for j in range(jobs):
                subrun_parameters = multi_runs[j*(job_parallelism):(j+1)*(job_parallelism)]
                subrun_name = run_names[j*(job_parallelism):(j+1)*(job_parallelism)]
                if len(subrun_parameters) == 1:
                    subrun_name = subrun_name[0]
                    subrun_parameters = subrun_parameters[0]
                run = ParallelRun(
                    subrun_parameters,
                    multi_gpu=False,
                    logger=grid_logger,
                    run_name=subrun_name,
                    slurm_script=slurm_script,
                    scheduler=scheduler,
                )
                run.launch(
                    only_create=only_create,
                    script_args=[
                        "--disable_log_params",
                        "--disable_log_on_file",
                        "--function",
                        function,
                    ],
                )
        else:
            for k, (subrun_parameters, subrun_name) in enumerate(
                zip(multi_runs, run_names)
            ):
                if len(multi_runs) > 1:
                    grid_logger.info(
                        f"Running subrun {k+1}/{len(multi_runs)} in run {i+1}/{len(runs_parameters)}"
                    )
                else:
                    grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
                run_function(subrun_parameters, run_name=subrun_name)


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--run_name", default=None, help="Name of the run")
@click.option(
    "--disable_log_params",
    default=False,
    is_flag=True,
    help="Disable Log the parameters",
)
@click.option(
    "--disable_log_on_file", default=False, is_flag=True, help="Disable Log on file"
)
@click.option(
    "--run_name",
    default=None,
    help="Name of the run, if not provided, it will be generated based on the current time",
)
@click.option(
    "--function",
    default="evaluate",
    help="Name of the function to run, either 'evaluate' or 'computational'",
)
def run(
    parameters,
    run_name=None,
    disable_log_params=False,
    disable_log_on_file=False,
    function="evaluate",
):
    assert function in FUNCTIONS, f"Function {function} not recognized, available functions: {list(FUNCTIONS.keys())}"
    run_function = FUNCTIONS[function]
    
    parameters = load_yaml(parameters)
    run_function(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_on_file,
    )

if __name__ == "__main__":
    cli()
