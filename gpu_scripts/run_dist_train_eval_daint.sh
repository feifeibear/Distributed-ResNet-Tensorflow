#!/bin/bash

# set TensorFlow distributed parameters
if [ -z "$TF_SCRIPT" ]; then
    echo "Set the variable TF_SCRIPT"
    exit 1
fi

if [ -z "$TF_NUM_PS" ]; then
    TF_NUM_PS=1
fi

let TF_NUM_WORKERS=${TF_NUM_WORKERS}-1 #$(($SLURM_JOB_NUM_NODES-$TF_NUM_PS))
if [ -z "$TF_NUM_WORKERS" ]; then
    TF_NUM_WORKERS=$SLURM_JOB_NUM_NODES #$(($SLURM_JOB_NUM_NODES-$TF_NUM_PS))
fi

if [ -z "$TF_WORKER_PER_NODE" ]; then
    TF_WORKER_PER_NODE=1
fi

if [ -z "$TF_PS_PER_NODE" ]; then
    TF_PS_PER_NODE=1
fi

if [ -z "$TF_PS_IN_WORKER" ]; then
    TF_PS_IN_WORKER=true
fi

# get PS and WORKER hostnames from Slurm
let num_ps_nodes=($TF_NUM_PS+$TF_PS_PER_NODE-1)/$TF_PS_PER_NODE
if [ $num_ps_nodes -gt $SLURM_JOB_NUM_NODES ]; then
    echo "The number of allocated nodes is not enough for your PS configuration"
    exit 1
fi
SLURM_PS_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                 head -n ${num_ps_nodes} | tr -s '\n' ',' | head --bytes -1)
i=0
for h in ${SLURM_PS_HOSTS//,/ }; do
  for (( j=0; j < ${TF_PS_PER_NODE}; j++ )) do
    if [ $i -lt ${TF_NUM_PS} ]; then
      TF_PS_HOSTS=$TF_PS_HOSTS"${h}:$((2230 + ${j})),"
      i=$((${i}+1))
    fi
  done
done
TF_PS_HOSTS=$( echo "$TF_PS_HOSTS" | head --bytes -2 )

let num_worker_nodes=($TF_NUM_WORKERS+$TF_WORKER_PER_NODE-1)/$TF_WORKER_PER_NODE
if [ "$TF_PS_IN_WORKER" = true ]; then
  if [ $num_worker_nodes -gt $SLURM_JOB_NUM_NODES ]; then
      echo "The number of allocated nodes is not enough for your WORKER setting"
      exit 1
  fi
  SLURM_WORKER_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       head -n ${num_worker_nodes} | tr -s '\n' ',' | 
                       head --bytes -1)
else
  if [ $num_worker_nodes -gt $(($SLURM_JOB_NUM_NODES-${TF_NUM_PS}-1)) ]; then
  # if [ $num_worker_nodes -gt $(($SLURM_JOB_NUM_NODES-$TF_NUM_PS)) ]; then
      echo "The number of allocated nodes is not enough for your WORKER setting"
      exit 1
  fi
  SLURM_WORKER_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       tail -n +$(( ${num_ps_nodes}+1 )) | 
                       head -n ${num_worker_nodes} | tr -s '\n' ',' | 
                       head --bytes -1)
fi
i=0
for h in ${SLURM_WORKER_HOSTS//,/ }; do
  for (( j=0; j < ${TF_WORKER_PER_NODE}; j++ )) do
    if [ $i -lt ${TF_NUM_WORKERS} ]; then
      TF_WORKER_HOSTS=$TF_WORKER_HOSTS"${h}:$((2220 + ${j})),"
      i=$((${i}+1))
    fi
  done
done
TF_WORKER_HOSTS=$( echo "$TF_WORKER_HOSTS" | head --bytes -2 )

# display actual paramemters
echo "--- Distributed TENSORFLOW ---"
echo "--- Number of PS: ${TF_NUM_PS}"
echo "--- Number of WORKERS: ${TF_NUM_WORKERS}"
echo "--- PS / node: ${TF_PS_PER_NODE}"
echo "--- WORKERS / node: ${TF_WORKER_PER_NODE}"
echo "--- PS hosts: ${TF_PS_HOSTS}"
echo "--- WORKER hosts: ${TF_WORKER_HOSTS}"
echo

TF_DIST_FLAGS=" --ps_hosts=${TF_PS_HOSTS} --worker_hosts=${TF_WORKER_HOSTS}"

# start PSs and WORKERs
running_nodes=0
if [ "$TF_PS_IN_WORKER" = true ]; then
  num_nodes=$((num_ps_nodes>num_worker_nodes ? num_ps_nodes : num_worker_nodes))
else
  num_nodes=$((num_ps_nodes + num_worker_nodes))
fi
IFS="," read -r -a ps_array <<< "$TF_PS_HOSTS"
IFS="," read -r -a w_array <<< "$TF_WORKER_HOSTS"
ps_task_index=0
w_task_index=0
ps_count=0
w_count=0
while [ $running_nodes -lt $num_nodes ]; do
  current_node=""
  current_node_ps=""
  current_node_w=""
  n_ps_current_node=0
  n_w_current_node=0

  # get PS hosts in node
  if [ $ps_count -lt $TF_NUM_PS ]; then
    # get node at position ps_count
    current_node=${ps_array[$ps_count]}
    current_node="$(cut -d':' -f1 <<<"$current_node")"

    # get all ps to be run on the current node
    for np in "${ps_array[@]:$ps_count}"; do
      if [[ ${np} == $current_node* ]]; then
        current_node_ps=$current_node_ps"${np},"
        n_ps_current_node=$((${n_ps_current_node}+1))
      else
        break
      fi
    done
    ps_count=$((${ps_count}+${n_ps_current_node}))
  fi

  # get WORKER hosts in node
  if [ $w_count -lt $TF_NUM_WORKERS ]; then
    if [ -z "$current_node" ]; then
      # no more PS -- get node at position w_count
      current_node=${w_array[$w_count]}
      current_node="$(cut -d':' -f1 <<<"$current_node")"
    fi

    # get all worker to be run on the current node
    for np in "${w_array[@]:$w_count}"; do
      if [[ ${np} == $current_node* ]]; then
        current_node_w=$current_node_w"${np},"
        n_w_current_node=$((${n_w_current_node}+1))
      else
        break
      fi
    done
    w_count=$((${w_count}+${n_w_current_node}))
  fi

  # write executable to be run on the current node
  slurm_node_script=.tfdist.${SLURM_JOBID}.${current_node}.sh
  echo "#!/bin/bash" > $slurm_node_script
  echo "cvd=\${CUDA_VISIBLE_DEVICES};CUDA_VISIBLE_DEVICES=" >> $slurm_node_script

  if  [ $n_ps_current_node -gt 0 ]; then
    for (( j=0; j < $((${n_ps_current_node}-1)); j++ )) do
      np="$(cut -d',' -f $(($j+1)) <<<"$current_node_ps")"
      echo "starting ps $ps_task_index: $np"
      PS_CMD="python3 ${TF_SCRIPT} --job_name=ps ${TF_DIST_FLAGS} 
              --task_index=${ps_task_index} ${TF_FLAGS} 
              > ps.${SLURM_JOBID}.${np//:/-}.log 2>&1 &" 
      echo ${PS_CMD} >> $slurm_node_script
      ps_task_index=$((${ps_task_index}+1))
    done
    np=$(cut -d',' -f $(($j+1)) <<<"$current_node_ps")
    echo "starting ps $ps_task_index: $np"
    if [ ${n_w_current_node} == 0 ]; then
      PS_CMD="python3 ${TF_SCRIPT} --job_name=ps ${TF_DIST_FLAGS} 
              --task_index=${ps_task_index} ${TF_FLAGS} 
              > ps.${SLURM_JOBID}.${np//:/-}.log 2>&1 "
    else
      PS_CMD="python3 ${TF_SCRIPT} --job_name=ps ${TF_DIST_FLAGS} 
              --task_index=${ps_task_index} ${TF_FLAGS} 
              > ps.${SLURM_JOBID}.${np//:/-}.log 2>&1 &"
    fi
    echo ${PS_CMD} >> $slurm_node_script
    ps_task_index=$((${ps_task_index}+1))
  fi 

  if  [ $n_w_current_node -gt 0 ]; then
    echo "sleep 10;CUDA_VISIBLE_DEVICES=\${cvd}" >> $slurm_node_script

    for (( j=0; j < $((${n_w_current_node}-1)); j++ )) do
      np=$(cut -d',' -f $(($j+1)) <<<"$current_node_w")
      echo "starting worker $w_task_index: $np"
      WORKER_CMD="python3 ${TF_SCRIPT} --job_name=worker ${TF_DIST_FLAGS} 
                  --task_index=${w_task_index} ${TF_FLAGS} 
                  > worker.${SLURM_JOBID}.${np//:/-}.log 2>&1 &"
      echo ${WORKER_CMD} >> $slurm_node_script
      w_task_index=$((${w_task_index}+1))
    done
    np=$(cut -d',' -f $(($j+1)) <<<"$current_node_w")
    echo "starting worker $w_task_index: $np"
    WORKER_CMD="python3 ${TF_SCRIPT} --job_name=worker ${TF_DIST_FLAGS} 
                --task_index=${w_task_index} ${TF_FLAGS} 
                > worker.${SLURM_JOBID}.${np//:/-}.log 2>&1 "
    echo ${WORKER_CMD} >> $slurm_node_script
    w_task_index=$((${w_task_index}+1))
  fi

  chmod +x $slurm_node_script
  if [ $((${running_nodes} + 1)) == $num_nodes ]; then
    srun --no-kill --nodelist $current_node -n 1 -N 1 $slurm_node_script & 
  else
    srun --no-kill --nodelist $current_node -n 1 -N 1 $slurm_node_script &
  fi
  running_nodes=$((${running_nodes} + 1))

done
#fjr add eval
SLURM_EVALER_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       tail -n 1 | tr -s '\n' ',' | 
                       head --bytes -1)
current_node=$SLURM_EVALER_HOSTS
TF_EVALER_HOSTS="${SLURM_JOB_NUM_NODES}:2220"
WORKER_CMD="python3 ${TF_EVAL_SCRIPT} ${TF_EVAL_FLAGS} > eval.${SLURM_JOBID}.${np//:/-}.log 2>&1"
slurm_node_script=.tfdist.${SLURM_JOBID}.${current_node}.sh
echo "#!/bin/bash" > $slurm_node_script
echo "cvd=\${CUDA_VISIBLE_DEVICES};CUDA_VISIBLE_DEVICES=\${cvd}" >> $slurm_node_script
echo ${WORKER_CMD} >> $slurm_node_script
chmod +x $slurm_node_script
srun --no-kill --nodelist $current_node -n 1 -N 1 $slurm_node_script


# rm .tfdist.${SLURM_JOBID}.*
# rm ps.${SLURM_JOBID}.*
# rm worker.${SLURM_JOBID}.*
