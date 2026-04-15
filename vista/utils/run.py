import os
import subprocess
import sys

from .utils import PrintLogger, write_yaml


class ParallelRun:
    slurm_command = "sbatch"
    slurm_multi_gpu_script = "slurm/launch_run_multi_gpu"
    slurm_script_first_parameter = "--parameters="
    slurm_script_run_name_parameter = "--run_name="
    out_extension = "log"
    param_extension = "yaml"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(
        self,
        params: dict,
        multi_gpu=False,
        logger=None,
        run_name=None,
        slurm_script=None,
        scheduler="slurm",
    ):
        self.params = params
        self.multi_gpu = multi_gpu
        self.logger = logger or PrintLogger()
        self.run_name = run_name
        self.slurm_script = slurm_script or "slurm/launch_run"
        self.scheduler = scheduler
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False, script_args=[]):
        if isinstance(self.params, list):
            self.launch_multi_task(only_create, script_args)
        else:
            self.launch_single_task(only_create, script_args)

    def launch_multi_task(self, only_create=False, script_args=[]):
        if self.scheduler != "slurm":
            raise NotImplementedError("Multi task launching is only implemented for slurm scheduler")
        self.logger.info(f"Running {len(self.params)} tasks in parallel")
        out_files = [f"{self.run_name[i]}/log.{self.out_extension}" for i in range(len(self.params))]
        
        for i in range(len(self.params)):
            write_yaml(self.params[i], f"{self.run_name[i]}/params.{self.param_extension}")
        
        # Load slurm script so that we can modify it for multi task
        with open(self.slurm_script, "r") as f:
            slurm_script = f.readlines()
            
        # Last line contains the execution file
        last_line = slurm_script[-1]
        if "srun" not in last_line:
            raise ValueError("Slurm script does not contain srun command for multi task execution")
        ex_file = last_line.split(" ")[1]
        
        ex_commands = []

        for i in range(len(self.params)):
            param_file = f"{self.run_name[i]}/params.{self.param_extension}"
            log_file = out_files[i]
            
            # Each task runs in background
            ex_command = (
                f"( {ex_file} {self.slurm_script_first_parameter}{param_file} "
                f"{self.slurm_script_run_name_parameter}{self.run_name[i]} "
                f"{' '.join(script_args)} $@ > {log_file} 2>&1 ) &"
            )
            ex_commands.append(ex_command)

        # Aggiungi il wait alla fine
        ex_commands.append("wait")

        # Combina tutto in un unico srun
        ex_commands = ["srun bash -c '" + " ".join(ex_commands) + "'\n"]
        
        slurm_script = slurm_script[:-1] + ex_commands
        
        # Create a temporary slurm script for multi task, run_names have this structure
        # out/xxx/job_<xxx>/p_<yyy> We put the slurm script in out/xxx/job_<xxx>/
        temp_slurm_script = os.path.join(os.path.dirname(self.run_name[0]), "multi_task_slurm.sh")
        with open(temp_slurm_script, "w") as f:
            f.writelines(slurm_script)
        # Job level log file
        job_log_file = os.path.join(os.path.dirname(self.run_name[0]), "job.log")
            
        command = [
            self.slurm_command,
            self.slurm_stdout,
            job_log_file,
            self.slurm_stderr,
            job_log_file,
            temp_slurm_script,
            *script_args,
        ]

        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)

    def launch_single_task(self, only_create=False, script_args=[]):
        out_file = f"{self.run_name}/log.{self.out_extension}"
        param_file = f"{self.run_name}/params.{self.param_extension}"

        self.logger.info("Running a single process")
        self.launch_process(
            self.params, self.run_name, out_file, param_file, only_create, script_args
        )

    def launch_process(
        self, params, run_name, out_file, param_file, only_create=False, script_args=[]
    ):
        write_yaml(params, param_file)
        slurm_script = (
            self.slurm_multi_gpu_script if self.multi_gpu else self.slurm_script
        )
        
        if self.scheduler == "slurm":
            self.launch_slurm(
                params, run_name, out_file, param_file, only_create, script_args
            )
        elif self.scheduler == "condor":
            self.launch_condor(
                params, run_name, out_file, param_file, only_create, script_args
            )
        else:
            raise ValueError(f"Scheduler {self.scheduler} not recognized")

    def launch_slurm(
        self, params, run_name, out_file, param_file, only_create=False, script_args=[]
    ):
        slurm_script = (
            self.slurm_multi_gpu_script if self.multi_gpu else self.slurm_script
        )
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            slurm_script,
            self.slurm_script_first_parameter + param_file,
            self.slurm_script_run_name_parameter + run_name,
            *script_args,
        ]

        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)

    def launch_condor(
        self, params, run_name, out_file, param_file, only_create=False, script_args=[]
    ):
        if self.multi_gpu:
            raise NotImplementedError("Multi GPU not implemented for condor scheduler")
        
        slurm_script = self.slurm_script
        
        command = [
            "condor_submit",
            f"output={out_file}",
            f"error={out_file}",
            f"log={out_file}",
            f"arguments='main.py run {self.slurm_script_first_parameter}{param_file} {self.slurm_script_run_name_parameter}{run_name} {' '.join(script_args)}'",
            slurm_script,
        ]
        
        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)

