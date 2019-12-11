#!/bin/bash

while getopts ":c:s:n" opt; do
  case $opt in
    c) CaP="$OPTARG"
    ;;
    s) JOBNAME="$OPTARG"
    ;;
    n) nbins="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

## give a name to your job
#SBATCH --job-name=JOBNAME

## your contact email
#SBATCH --mail-user=roman.koshkin@oist.jp

## number of cores for your simulation,
## for serial job array it is always 1
#SBATCH --ntasks=1

#SBATCH --partition=compute

## how much memory per core
#SBATCH --mem-per-cpu=4g

## submit 4 jobs as an array, give them individual id from 1 to 4
#SBATCH --array=1-20

## maximum time for your simulation, in DAY-HOUR:MINUTE:SECOND
#SBATCH --time=0-0:20:0

## source the prebuilt STEPS environment and installation
## note that this is bulit with Python3.5
source /apps/unit/DeSchutterU/steps_env/2019_10/profiles/default


## run serial STEPS simulations, use $SLURM_ARRAY_TASK_ID as input
## to generate different output files

## WARNING!!!!: it is very important to store your data in different output files
## bacause each job in the array will write data to file simultaneously
## the following example script will generate 4 output files
## 1.out ~ 4.out
python Ca_Buffer_GHK_ser.py $SLURM_ARRAY_TASK_ID $CaP $JOBNAME $nbins