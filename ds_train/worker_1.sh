#!/bin/bash

if [ ! -d data ]; then
  ln -s /data/data1/v-qizhe/data data
fi

#export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="2"

python train.py \
    --mode="train"\
    --src_file_name="data/train.en"\
    --tgt_file_name="data/train.de"\
    --src_vocab_file="data/vocab.50K.en"\
    --tgt_vocab_file="data/vocab.50K.de"\
    --batch_size=256 \
    --variable_update="parameter_server" \
    --dropout=0.4 \
    --unit_type="lstm" \
    --num_units=1024 \
    --beam_width=10 \
    --forget_bias=0.8 \
    --embedding_dim=1024 \
    --src_max_len=50 \
    --tgt_max_len=50 \
    --num_encoder_layers=4 \
    --num_decoder_layers=4 \
    --encoder_type="bi" \
    --max_gradient_norm=5.0 \
    --learning_rate 0.001 \
    --use_attention=True \
    --task_index=1 \
    --job_name="worker" \
#    --use_residual_connection="True" \
#    --prefetch_data_to_device="true" \
