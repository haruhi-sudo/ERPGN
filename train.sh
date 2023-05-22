TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/model/aruhi/bart_base/bart.base/model.pt
# fairseq-train

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    /data/aruhi/cnn_dailymail/cnn-dm-base \
    --max-epoch 3 \
    --user-dir  src \
    --save-dir  /model/aruhi/bart_base/cnn-dm\
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task faithful_summarization \
    --source-lang source --target-lang target \
    --reset-optimizer --reset-dataloader --reset-meters \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_pg_base\
    --fp16 --criterion my_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters
