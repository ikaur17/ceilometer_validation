#!/bin/bash -l                                                                                                    
#SBATCH -A safnwc
#SBATCH -n 12 # lower case n!
#SBATCH -t 20:00:0
#SBATCH --array=0-332%10  

module load  Mambaforge/23.3.1-1-hpc1
conda activate cbase
export PYTHONPATH=/home/sm_indka/Projects/ceilometer_validation/cm_validation/:$PYTHONPATH

python runner_avg.py ${SLURM_ARRAY_TASK_ID}
