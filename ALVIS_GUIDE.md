# Alvis tutorial

This tutorial is based on the [Alvis tutorial](https://www.c3se.chalmers.se/documentation/intro-alvis/slides/).



### General resources
* [Alvis](https://www.c3se.chalmers.se/about/Alvis/)
* [SLURM](https://slurm.schedmd.com/documentation.html)

### Logging in

An example of how to log in to Alvis:

```bash
ssh CID@alvis1.c3se.chalmers.se
```

If you are not connected to Chalmers internet, you need to use a Chalmers VPN.


### Storage

The command:
    
 ```bash
C3SE_quota
 ```

will show you the amount of storage you have left. And be aware of you storage quota!

For downloading and uploading files, you can use `scp`.


### Usage

* `projinfo`
* System status
* `jobinfo -u lovhag`
* Connect with [VS code](https://www.c3se.chalmers.se/documentation/remote-vscode/remote_vscode/)
* "Remote Explorer"
* Capacity of GPUs
* Memory overload "CUDA out of memory" "OOM"

### Environment
* Modules
* `python3 -m venv <path>`
* Use symbolic links!


### Starting a job script

```bash
#!/usr/bin/env bash

#SBATCH -A SNIC2022-22-1040 -p alvis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=FiD-NQ-eval
#SBATCH -o /cephyr/users/lovhag/Alvis/projects/SKR-project/FiD/NQ-eval.out
#SBATCH -t 4:00:00

set -eo pipefail

module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.7.1-fosscuda-2020b
source venv/bin/activate

DATA_ROOT="/cephyr/NOBACKUP/groups/snic2021-23-309/project-data/FiD"

python test_reader.py \
--model_path "${DATA_ROOT}/models/nq_reader_base" \
--eval_data "${DATA_ROOT}/NQ/test.json" \
--per_gpu_batch_size 8 \
--n_context 100 \
--name NQ_test_base_batch_8 \
--checkpoint_dir "${DATA_ROOT}/test_checkpoints" \
--write_results \
```

`job_stats.py JOBID`

Make sure that it is runnable!

Monitor with scruffy:
"This job can be monitored from: https://scruffy.c3se.chalmers.se/d/alvis-job/alvis-job?var-jobid=680110&from=1669896709000"

Make sure that all allocated resources are used

`sbatch jobscript`

`scancel`

## Job command overview
- `sbatch`: submit batch jobs
- `srun`: submit interactive jobs
- `jobinfo (squeue)`: view the job-queue and the state of jobs in queue, shows amount of idling resources
- `scontrol show job <jobid>`: show details about job, including reasons why it's pending
sprio: show all your pending jobs and their priority
- `scancel`: cancel a running or pending job
sinfo: show status for the partitions (queues): how many nodes are free, how many are down, busy, etc.
- `projinfo`: show the projects you belong to, including monthly allocation and usage
For details, refer to the -h flag, man pages, or google!



### Running a notebook

```bash
srun -p alvis -A SNIC2022-22-1040 -N 1 --gpus-per-node=T4:1 --job-name=demo -t 4:00:00 --pty bash
module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.8.1-fosscuda-2020b torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1
source ../venv/bin/activate

jupyter notebook ../data/larger-visual-commonsense-eval-experiments/normdata-evaluation-results.ipynb
```

Read more here: [https://www.c3se.chalmers.se/documentation/applications/jupyter/](https://www.c3se.chalmers.se/documentation/applications/jupyter/)

### Typical issues

* Logfile
* Environment
* Storage full
* What to do if something doesn't work
* Read docs

Contact Lovisa if anything is unclear or if you have any questions!