export PRED_FILE=sample/Jaco126.fa
export PRE_WEIGHT=pretrained/bert_mul_2.pth
export OUTPUT_FILE=output/Jaco126.emb

python MLM_SFP.py \
    --pretraining ${PRE_WEIGHT} \
    --data_embedding ${PRED_FILE} \
    --embedding_output ${OUTPUT_FILE} \
    --batch 40 \

#    --pretraining pretrained/bert_mul_2.pth --data_embedding sample/Jaco126.fa --embedding_output output/Jaco126.emb --batch 36
