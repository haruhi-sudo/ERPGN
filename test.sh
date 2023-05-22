python summarize.py \
  --model-dir /model/aruhi/bart_base/cnn-dm \
  --model-file checkpoint3.pt \
  --src /data/aruhi/cnn_dailymail/test.source \
  --out /output/test.txt \
  --entity-file /data/aruhi/cnn_dailymail/cnn-dm-base/test.entity.source \
  --triple-file /data/aruhi/cnn_dailymail/cnn-dm-base/test.bpe.triples.source \
#   --xsum-kwargs
