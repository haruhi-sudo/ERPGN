import copy
import logging
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict


logger = logging.getLogger(__name__)


class BPHubInterface(GeneratorHubInterface):

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]
        self.max_positions = (512, 512)

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        bpe_sentence = tokens + " </s>"
        # bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_batches(
        self, tokens: List[List[int]], entity_group, triples_dataset, skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths, triples_dataset, entity_group),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def sample(
        self, sentences: List[str], entity_group, triples_dataset, beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        triples_dataset = [self.bpe.decode(one_triples_dataset) for one_triples_dataset in triples_dataset]
        triples_dataset = [self.encode(one_triples_dataset) for one_triples_dataset in triples_dataset]
        batched_hypos = self.generate(tokenized_sentences, entity_group, triples_dataset, beam, verbose, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        entity_group,
        triples_dataset,
        *args,
        inference_step_args=None,
        skip_invalid_size_inputs=False,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        inference_step_args = inference_step_args or {}
        res = []

        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = args[0]
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            prefix_allowed_tokens_fn=None,
        )
        for batch in self._build_batches(tokenized_sentences, entity_group, triples_dataset, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                res.append((id, hypos))

            # sort output to match input order
            outputs = [hypos for _, hypos in sorted(res, key=lambda x: x[0])]

        return outputs

