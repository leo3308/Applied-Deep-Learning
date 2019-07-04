
export DATA_DIR=../data/

python3.7 run_classifier.py \
    --task_name="mytask" \
    --bert_model="bert-base-cased" \
    --do_predict \
    --warmup_proportion=0.1 \
    --data_dir=${1} \
    --max_seq_length=64 \
    --train_batch_size=32 \
    --eval_batch_size=8 \
    --num_train_epochs 3 \
    --output_dir="predict_base_cased" \
    --output_path=${2} \
    --seed=143 \
    --learning_rate 2e-5 
