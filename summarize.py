import torch
from src.model.bart_pg import BARTPointModel 
import argparse
import json

XSUM_KWARGS = dict(beam=8, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
CNN_KWARGS = dict(beam=15, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

def load_entity_dataset(data_path):
    with open(data_path, 'r') as f:
        entity_group = json.load(f)
    return entity_group

def load_triples_dataset(data_path):
    triples = []
    with open(data_path, 'r') as f:
        for triple in f.readlines():
            triples.append(triple)
    return triples


@torch.no_grad()
def generate(bart, infile, entityfile, triplefile, outfile="bart_hypo.txt", bsz=32, n_obs=None, **eval_kwargs):
    count = 1

    # if n_obs is not None: bsz = min(bsz, n_obs)
    entity_group = load_entity_dataset(entityfile)
    triples = load_triples_dataset(triplefile)
    with open(infile) as source, open(outfile, "w") as fout:
        sline = ' ' + source.readline().strip()
        slines = [sline]
        entity_group_batch = [entity_group[0]]
        triples_batch = [triples[0].strip()]
        for sline in source:
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                hypotheses_batch = bart.sample(slines,entity_group_batch,triples_batch,**eval_kwargs)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []
                entity_group_batch = []
                triples_batch = []

            slines.append(' ' + sline.strip())
            entity_group_batch.append(entity_group[count])
            triples_batch.append(triples[count].strip())
            count += 1

        if slines != []:
            hypotheses_batch = bart.sample(slines, entity_group_batch, triples_batch, **eval_kwargs)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()


def main():
    """
    Usage::
         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument(
        "--xsum-kwargs",
        action="store_true",
        default=False,
        help="if true use XSUM_KWARGS else CNN_KWARGS",
    )
    parser.add_argument(
        "--entity-file", default="test.entity.source", help="entity", type=str
    )
    parser.add_argument(
        "--triple-file", default="test.triples.source", help="entity", type=str
    )
    args = parser.parse_args()
    eval_kwargs = XSUM_KWARGS if args.xsum_kwargs else CNN_KWARGS
    if args.model_dir == "pytorch/fairseq":
        bart = torch.hub.load("pytorch/fairseq", args.model_file)
    else:
        bart = BARTPointModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
    bart = bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda()
    generate(
        bart, args.src, entityfile=args.entity_file, triplefile=args.triple_file, 
         bsz=args.bsz, n_obs=args.n, outfile=args.out, **eval_kwargs
    )


if __name__ == "__main__":
    main()