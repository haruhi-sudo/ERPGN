from dataclasses import dataclass, field
import json
import logging
import os
import torch
from typing import Optional
from argparse import Namespace
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.gpt2_bpe import get_encoder
from .entity_dataset import EntityDataset

bpe = get_encoder('encoder.json', 'vocab.bpe')
logger = logging.getLogger(__name__)

def load_entity_dataset(data_path):
    with open(data_path, 'r') as f:
        entity_group = json.load(f)
    return entity_group


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    truncate_source=False,
    shuffle=True,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=None)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
    else:
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, data_path)
        )

    src_dataset = data_utils.load_indexed_dataset(
        prefix + src, src_dict
    )

    src_entity_dataset = load_entity_dataset(
        os.path.join(data_path, split+'.entity.'+src)
    )
    triples_dataset = data_utils.load_indexed_dataset(
        os.path.join(data_path, split+'.source-None.'+src), src_dict
    )
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )
        triples_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(triples_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )

    tgt_dataset = data_utils.load_indexed_dataset(
        prefix + tgt, tgt_dict
    )
    tgt_entity_dataset =  load_entity_dataset(
        os.path.join(data_path, split+'.entity.'+tgt)
    )

    logger.info(
        "{} {} {}-{} {} examples".format(
            data_path, split, src, tgt, len(src_dataset)
        )
    )

    src_dataset_sizes = src_dataset.sizes
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return EntityDataset(
        src_dataset,
        src_dataset_sizes,
        src_dict,
        src_entity_dataset,
        triples_dataset, 
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        tgt_entity_dataset,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        shuffle=shuffle,
    )


@dataclass
class SummarizationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=True, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default='{"beam":6, "lenpen":1.0, "max_len_b":60, "min_len":10, "no_repeat_ngram_size":3}',
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("faithful_summarization", dataclass=SummarizationConfig)
class FaithfulSummarizationTask(FairseqTask):
    cfg: SummarizationConfig

    def __init__(self, cfg: SummarizationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: SummarizationConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[0]
        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            truncate_source=self.cfg.truncate_source,
            shuffle=(split != "test"),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, triples_dataset, src_entity_dataset):
        return EntityDataset(
            src_tokens,
            src_lengths,
            triples_dataset = triples_dataset,
            src_entity_dataset=src_entity_dataset,
            src_dict=self.source_dictionary,
            tgt_dict=self.target_dictionary,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        if self.cfg.eval_bleu:
            metrics = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["ROUGE_L"] = metrics['rouge-l']['f']
            logging_output["ROUGE_1"] = metrics['rouge-1']['f']
            logging_output["ROUGE_2"] = metrics['rouge-2']['f']
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result
            
            metrics.log_scalar("ROUGE_1", sum_logs("ROUGE_1"))
            metrics.log_scalar("ROUGE_2", sum_logs("ROUGE_2"))
            metrics.log_scalar("ROUGE_L", sum_logs("ROUGE_L"))

    def max_positions(self):
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs, articles = [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
            articles.append(
                decode(
                    utils.strip_pad(sample["net_input"]["src_tokens"][i], self.tgt_dict.pad()),
                    escape_unk=True,
                )
            )
        if self.cfg.eval_bleu_print_samples:
            for hyp, article in zip(hyps, articles):
                logger.info("example hypothesis: " + bpe.decode([int(tok) for tok in hyp.split(' ')]))
                logger.info("example article: " + bpe.decode([int(tok) for tok in article.split(' ')]))

        from rouge import Rouge
        rouge = Rouge()
        metrics = rouge.get_scores(hyps, refs, avg=True)
        return metrics
