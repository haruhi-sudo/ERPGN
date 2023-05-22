import torch
import logging
from fairseq.data import data_utils, LanguagePairDataset
from fairseq.data.encoders.gpt2_bpe import get_encoder

logger = logging.getLogger(__name__)
bpe = get_encoder('encoder.json', 'vocab.bpe')

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        pad_idx=pad_idx,
        left_pad=left_pad_source,
    )
    src_entity_mask = merge(
        "src_entity_mask",
        pad_idx=1,
        left_pad=left_pad_source,
    )
    src_triples = merge(
        "src_triples",
        pad_idx=pad_idx,
        left_pad=left_pad_source,
    )
    src_entity = [s["src_entity"] for s in samples]
    
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_triples_lengths = torch.LongTensor(
        [s["src_triples"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_triples_lengths = src_triples_lengths.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_triples = src_triples.index_select(0, sort_order)
    src_entity_mask = src_entity_mask.index_select(0, sort_order)
    src_entity = [src_entity[i] for i in sort_order]

    prev_output_tokens = None
    target = None
    tgt_entity_mask = None
    tgt_entity = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            pad_idx=pad_idx,
            left_pad=left_pad_target,
        )
        tgt_entity_mask = merge(
            "tgt_entity_mask",
            pad_idx=1,
            left_pad=left_pad_target,
        )
        tgt_entity = [s["tgt_entity"] for s in samples]
        
        tgt_entity_mask = tgt_entity_mask.index_select(0, sort_order)
        tgt_entity = [tgt_entity[i] for i in sort_order]
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", pad_idx=pad_idx, left_pad=left_pad_target)
        elif input_feeding:
            prev_output_tokens = merge(
                "target",
                pad_idx=pad_idx,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_triples": src_triples,
            "src_triples_lengths": src_triples_lengths,
            "src_entity_mask": src_entity_mask,
        },
        "target": target,
        "src_entity": src_entity,
        "tgt_entity": tgt_entity,
        "tgt_entity_mask": tgt_entity_mask,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )
    # bpe.decode([int(tok) for tok in dict_.string(batch['net_input']['src_tokens'][1].numpy()).split(' ') if tok != '<pad>'])
    return batch


class EntityDataset(LanguagePairDataset):
    def __init__(
        self,
        src_dataset,
        src_dataset_sizes,
        src_dict,
        src_entity_dataset,
        triples_dataset,
        tgt_dataset=None,
        tgt_dataset_sizes=None,
        tgt_dict=None,
        tgt_entity_dataset=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
    ):
        super().__init__(
            src_dataset,
            src_dataset_sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            shuffle=shuffle,
        )
        self.src_entity = src_entity_dataset
        self.src_triples = triples_dataset
        self.tgt_entity = tgt_entity_dataset


    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        tgt_entity_item = self.tgt_entity[index] if self.tgt_entity is not None else None
        src_item = self.src[index]
        src_entity_item = self.src_entity[index]
        src_triples_item = self.src_triples[index]
        
        # TODO fixbug
        src_entity_mask = []
        idx = 0
        for tok in self.src_dict.string(src_item.numpy()).split(' '):
            if tok == '<pad>':
                src_entity_mask.append(1)
                continue
            if idx >= len(src_entity_item):
                src_entity_mask.append(1)
                continue
            # ' Smith' == 'ĠSmith', 'Smith' == 'Smith', 'a' != 'z'
            if bpe.decode([int(tok)]) == src_entity_item[idx]['word'] or \
              bpe.decode([int(tok)])[1:] == src_entity_item[idx]['word'][1:] \
              and src_entity_item[idx]['word'][1:] != None:
                src_entity_mask.append(0)
                idx += 1
            else:
                src_entity_mask.append(1)
        
        src_entity_mask.append(1)
        assert len(src_entity_mask) == len(src_item)

        tgt_entity_mask = None
        if self.tgt is not None:
            tgt_entity_mask = []
            idx = 0
            for tok in self.tgt_dict.string(tgt_item.numpy()).split(' '):
                if tok == '<pad>':
                    tgt_entity_mask.append(1)
                    continue
                if idx >= len(tgt_entity_item):
                    tgt_entity_mask.append(1)
                    continue
                # ' Smith' == 'ĠSmith', 'Smith' == 'Smith', 'a' != 'z'
                if bpe.decode([int(tok)]) == tgt_entity_item[idx]['word'] or \
                  bpe.decode([int(tok)])[1:] == tgt_entity_item[idx]['word'][1:] \
                  and tgt_entity_item[idx]['word'][1:] != None:
                    tgt_entity_mask.append(0)
                    idx += 1
                else:
                    tgt_entity_mask.append(1)
        
            tgt_entity_mask.append(1)
            assert len(tgt_entity_mask) == len(tgt_item)

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "src_entity": src_entity_item,
            "src_entity_mask": torch.LongTensor(src_entity_mask),
            "src_triples": torch.LongTensor(src_triples_item),
            "tgt_entity_mask": torch.LongTensor(tgt_entity_mask) if tgt_entity_mask is not None else None,
            "tgt_entity": tgt_entity_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
        
        return res

