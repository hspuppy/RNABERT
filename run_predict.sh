export PRED_FILE=sample/aln/sample.raw.fa
export PRE_WEIGHT=pretrained/bert_mul_2.pth

python MLM_SFP.py \
    --pretraining ${PRE_WEIGHT} \
    --data_alignment ${PRED_FILE} \
    --batch 40 \
    --show_aln
