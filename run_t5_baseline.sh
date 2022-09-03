# 设置GPU可见，即指定使用GPU的卡号
export CUDA_VISIBLE_DEVICES=5

# 根路径
ROOT_DIR=$(pwd)/data
# 任务路径
TASK_DIR=${ROOT_DIR}/ner
# 预训练模型类型
MODEL_TYPE="t5-base-en"
# 预训练模型路径
PRE_TRAINED_MODEL_DIR=${ROOT_DIR}/pretrain_model/${MODEL_TYPE}/
# 微调模型存储路径
FINETUNE_MODEL_DIR=${TASK_DIR}/model/t5_baseline
FINETUNE_MODEL_PATH=${FINETUNE_MODEL_DIR}/t5_prompt_ner.pk

# 日志文件夹
LOG_DIR=log

# 创建相关目录
mkdir -p ${LOG_DIR}
mkdir -p ${FINETUNE_MODEL_DIR}

# 日志文件
#LOG_FILE=${LOG_DIR}/test.txt
LOG_FILE=${LOG_DIR}/t5_prompt_train_ace2004.txt
#LOG_FILE=${LOG_DIR}/t5_note_train_cybersecurity.txt

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
#FORMAT_DATA_DIR=${ROOT_DIR}/datasets/DNRTI/format
#FORMAT_DATA_DIR=${ROOT_DIR}/datasets/ACE2004/format

# prompt
FORMAT_DATA_DIR=${ROOT_DIR}/datasets/ACE2004/retrieval
#FORMAT_DATA_DIR=${ROOT_DIR}/datasets/SHARE13/retrieval
TRAIN_DATA_PATH=${FORMAT_DATA_DIR}/train.txt
DEV_DATA_PATH=${FORMAT_DATA_DIR}/valid.txt
TEST_DATA_PATH=${FORMAT_DATA_DIR}/test.txt

# nohup将文件运行结果写入指定文件
nohup python -u run_model/run_t5_baseline.py \
  --prompt=True \
  --do_train \
  --pretrain_model_path=${PRE_TRAINED_MODEL_DIR} \
  --output_dir=${FINETUNE_MODEL_DIR} \
  --model_save_path=${FINETUNE_MODEL_PATH} \
  --train_data_path=${TRAIN_DATA_PATH} \
  --dev_data_path=${DEV_DATA_PATH} \
  --test_data_path=${TEST_DATA_PATH} \
  --dataloader_proc_num=4 \
  --epoch_num=100 \
  --per_device_train_batch_size=128 \
  --per_device_eval_batch_size=128 \
  --eval_batch_step=30 \
  --require_improvement_step=10000 \
  --max_input_len=100 \
  --max_target_len=50 \
  --beam_num=1 \
  --ignore_pad_token_for_loss=true \
  --pad_to_max_length=true \
  --learning_rate=5e-5 \
  --weight_decay=0.01 \
  --warmup_ratio=0.1 \
  --seed=42 \
  > ${LOG_FILE} 2>&1 &
