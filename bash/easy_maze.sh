#!/bin/bash -l

eval $1
eval $2
eval $3

jid_list=()

for job in ${job_list[*]}
do
    if [ $job == "break" ]
    then
        :
    elif [ $job == "empty_space" ]
    then
        :
    else
        singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --arg_title $job
        jid=$(sbatch --array=1-${agents} easy_maze/bash/main_$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --arg_title $job --post True
        jid=$(sbatch --dependency afterok:$jid easy_maze/bash/post_$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid_list+=($jid)
        #rm easy_maze/bash/$job.slurm
    fi
done

first_job=true
order="___"
for job in ${job_list[*]}
do 
    if [ $first_job = false ] 
    then
        order+="+"
    fi
    order+="$job" 
    first_job=false
done
order+="___"

singularity exec t_maze.sif python easy_maze/bash/slurmcraft.py --comp ${comp} --arg_title $order --post True
jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) easy_maze/bash/post_final.slurm)
echo $jid