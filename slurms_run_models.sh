#!/usr/bin/env bash

OUTDIR='/scratch/opt/tle/FluidDynamicsSimulationChallenge'

sbatch --job-name=Instability --time=24:00:00 --mem=128000 --cpus-per-task=4 --output=${OUTDIR}/job_%j.log /scratch/opt/tle/FluidDynamicsSimulationChallenge/run_models_slurm.sh

