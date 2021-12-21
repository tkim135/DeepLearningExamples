#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-128}
learning_rate=${2:-"0.00015"}
beta1=${22:-"0.9"}
beta2=${23:-"0.999"}
precision=${3:-"fp16"}
num_gpus=${4:-4}
warmup_proportion=${5:-"0.01"}
train_steps=${6:-300000}
save_checkpoint_steps=${7:-5000}
resume_training=${8:-"false"}
resume_step=${21:-"0"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-64}
seed=${12:-12439}
job_name=${13:-"wwmmlm_gpt2small_ss1024_maxpred320_300k_bs128_fp16"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
DATASET=wikienglish_wwm_ss1024_maxpred320
init_checkpoint=${17:-"None"}
master_port=${18:-9903}
BERT_CONFIG=gpt2small_mlm.json
DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
CODEDIR=${19:-"/workspace/bert"}
#DATA_DIR_PHASE1=${20:-$BERT_PREP_WORKING_DIR/${DATASET}/}
DATA_DIR_PHASE1=$CODEDIR/wikicorpus_data/wikibooks_wwm_ss1024_maxpred320/wikicorpus_books/
RESULTS_DIR="results"
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints_${job_name}
steps_this_run=${21:-300000}
log_freq=${22:-1}
weight_decay=${23:-0.1}

mkdir -p $CHECKPOINTS_DIR

if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
   echo "ACCUM GRADS ${ACCUMULATE_GRADIENTS}" 
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   echo "RESUME FROM CHECKPOINT"
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   echo "INIT CHECKPOINTS $init_checkpoint"
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" run_pretrain_gpt2.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=1024"
CMD+=" --max_predictions_per_seq=160"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --steps_this_run=$steps_this_run"
CMD+=" --resume_step=$resume_step"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --beta1=$beta1"
CMD+=" --beta2=$beta2"
CMD+=" --seed=$seed"
CMD+=" --log_freq=$log_freq"
CMD+=" --weight_decay=$weight_decay"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=$master_port $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  echo "GLOBAL BATCH SIZE ${GBS}"
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining"

