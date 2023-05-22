import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.data import data_utils
from src.model.discriminator import RCDiscriminator


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    rc_dis: str = field(
        default='.',
        metadata={"help": "relation consistency discriminator"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "my_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        rc_dis,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.rc_dis = RCDiscriminator(51200, 768)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        tokens_probs_differentiable = net_output[1]['tokens_probs_differentiable']

        entity_probs_max = net_output[1]["entity_probs_max"] + 1e-9
        entity_loss = - entity_probs_max.log().masked_fill(sample["tgt_entity_mask"]==1, 0).view(-1,).sum()

        src_triples_split = []
        iidx = 0
        tokens_probs_differentiable_spilt = []
        for triples in sample["net_input"]["src_triples"].cpu().numpy():
            tmp = []
            for tok in triples:
                if tok == 1: # 1: pad
                    continue
                elif tok == 50118: # 50118: \n
                    # tmp.append(tok)
                    tmp.append(2)
                    src_triples_split.append(torch.LongTensor(tmp))
                    tokens_probs_differentiable_spilt.append(tokens_probs_differentiable[iidx])
                    tmp = []
                else:
                    tmp.append(tok)
            iidx += 1
        
        assert len(tokens_probs_differentiable_spilt) == len(src_triples_split)
        
        self.rc_dis.eval()
        if len(src_triples_split) != 0:
            src_triples_split = data_utils.collate_tokens(src_triples_split, 1, left_pad=True)
            tokens_probs_differentiable_spilt = torch.stack(tokens_probs_differentiable_spilt)

            log_probs = self.rc_dis(
                tokens_probs_differentiable_spilt, 
                src_triples_split.to(tokens_probs_differentiable_spilt.device)
           )
            # except irrelevant type
            # condition = (log_probs[:,2] > log_probs[:,1]) & (log_probs[:,2] > log_probs[:,1])
            # 0: consistent
            # relation_loss = - torch.where(condition, torch.zeros_like(log_probs[:,0]), log_probs[:,0]).sum()
            relation_loss = log_probs[:,1].sum()
        else:
           relation_loss = torch.tensor(0).cuda()

        loss_lm, nll_loss_lm = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = 0.7 * loss_lm + 0.1 * entity_loss + 0.2 * relation_loss
        # loss = 0.7 * loss_lm + 0.3 * entity_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "loss_lm": loss_lm.data,
            "nll_loss_lm": nll_loss_lm.data,
            "entity_loss": entity_loss.data,
            "relation_loss": relation_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_lm_sum = sum(log.get("loss_lm", 0) for log in logging_outputs)
        nll_loss_lm_sum = sum(log.get("nll_loss_lm", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        entity_loss = sum(log.get("entity_loss", 0) for log in logging_outputs)
        relation_loss = sum(log.get("relation_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_lm", loss_lm_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "entity_loss", entity_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "relation_loss", relation_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss_lm", nll_loss_lm_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss_lm"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
