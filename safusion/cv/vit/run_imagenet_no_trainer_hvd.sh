

OUT_DIR=${OUT_DIR:-"./log"}
num_train_epochs="${num_train_epochs:-80000}"
density="${density:-0.1}"

compressor="${compressor:-dgc}"
# memory="${memory:-none}"
# memory="${memory:-residual}"
threshold="${threshold:-8192}"
percent="${percent:-0}"

per_device_train_batch_size="${per_device_train_batch_size:-32}"
per_device_eval_batch_size="${per_device_eval_batch_size:-32}"

# beans
# train_dir=${train_dir:-"/data/beans/train"}
# validation_dir=${validation_dir:-"/data/beans/validation"}

# # imagenet
train_dir=${train_dir:-"/data/imagenet/train"}
validation_dir=${validation_dir:-"/data/imagenet/val"}

# ViT-base
model_name_or_path=${model_name_or_path:-"/data/google/vit-base-patch16-224-in21k"}

metric_accuracy=${metric_accuracy:-"/data/google/evaluate/metrics/accuracy"}



output_dir=${output_dir:-"/data/image-classification/output"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi


export save_checkpoint_path="./horovod/example/elastic/pytorch/nlp/gpt/language-modeling/gpt2_checkpoint"




CMD=" HOROVOD_GPU_OPERATIONS=NCCL  HOROVOD_CACHE_CAPACITY=0 "

# CMD=" accelerate launch run_imagenet_no_trainer_hvd.py --image_column_name image   "
CMD=" horovodrun  -np  8 -H node15:2,node16:2,node19:2,node20:2  python run_imagenet_no_trainer_hvd.py --image_column_name image   "



CMD+=" --image_column_name image  "
CMD+=" --num_train_epochs=$num_train_epochs  "

CMD+=" --train_dir $train_dir  "
CMD+=" --validation_dir $validation_dir  "
CMD+=" --model_name_or_path $model_name_or_path  "
CMD+=" --per_device_train_batch_size $per_device_train_batch_size  "
CMD+=" --per_device_eval_batch_size $per_device_eval_batch_size  "
CMD+=" --metric_accuracy $metric_accuracy  --with_tracking "

CMD+=" --output_dir $output_dir  "


# CMD+=" --image_column_name image  "
# CMD+=" --image_column_name image  "


# CMD+=" --dataset_name /data/dataset/nlp/openai-community/wikitext-103-raw-v1 --dataset_config_name default  "
# CMD+=" --dataset_name /data/dataset/nlp/openai-community/wikitext-2-raw-v1 --dataset_config_name default "
# CMD+=" --model_name_or_path /data/dataset/nlp/openai-community/gpt2 "
# CMD+=" --output_dir  ./gpt2_checkpoint/ "
# CMD+=" --num_train_epochs=$epochs  "
# CMD+=" --density=$density --compressor=$compressor --memory=$memory --percent=$percent "
# CMD+=" --per_device_train_batch_size=$train_batch_size "
# CMD+=" --per_device_eval_batch_size=$val_batch_size "
# CMD+=" --resume_from_checkpoint  $Save_Checkpoint"


LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE






