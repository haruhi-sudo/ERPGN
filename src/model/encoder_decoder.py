import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder
)
from typing import Any, Dict, Optional, List, Tuple
from torch import Tensor
import torch.nn.functional as F


class TransformerPointerGeneratorEncoder(TransformerEncoder):
    def forward(
        self,
        src_tokens,
        src_entity_mask,
        src_triples,
        src_triples_lengths: Optional[Tensor] = None,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None
    ):
        encoder_out = self.forward_scriptable(src_tokens,
                                              src_lengths,
                                              return_all_hiddens,
                                              token_embeddings)
        triples_encoder_out = self.forward_scriptable(src_triples,
                                              src_triples_lengths,
                                              return_all_hiddens,
                                              token_embeddings)
        encoder_out = {
            "encoder_out": encoder_out["encoder_out"],  # T x B x C
            "encoder_padding_mask": encoder_out["encoder_padding_mask"],  # B x T
            # "encoder_embedding": encoder_out["encoder_embedding"],  # B x T x C
            # "encoder_states": encoder_out["encoder_states"],  # List[T x B x C]
            "src_tokens": [src_tokens],  # B x T
            "src_entity_mask": [src_entity_mask],
            # "src_lengths": [src_lengths],
        }
        triples_encoder_out = {
            "encoder_out": triples_encoder_out["encoder_out"],  # T x B x C
            "encoder_padding_mask": triples_encoder_out["encoder_padding_mask"],
            "src_tokens": [src_triples],  # B x T
        }

        return [encoder_out, triples_encoder_out]
        

    def reorder_encoder_out(self, encoder_outs: List[Dict[str, List[Tensor]]], new_order):
        encoder_out = encoder_outs[0]
        triples_encoder_out = encoder_outs[1]
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        new_encoder_padding_mask = [
            encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
        ]

        src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]
        src_entity_mask = [(encoder_out["src_entity_mask"][0]).index_select(0, new_order)]
        
        new_triples_encoder_padding_mask = [
            triples_encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
        ]

        new_triples_encoder_out = [triples_encoder_out["encoder_out"][0].index_select(1, new_order)]
        new_triples_src_tokens = [(triples_encoder_out["src_tokens"][0]).index_select(0, new_order)]

        return [
        {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,
            "src_tokens": src_tokens,  # B x T
            "src_entity_mask": src_entity_mask,  # B x 1
        }, 
        {
            "encoder_out": new_triples_encoder_out,
            "encoder_padding_mask":new_triples_encoder_padding_mask,
            "src_tokens":new_triples_src_tokens,
        }
        ]

    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)


class TransformerPointerGeneratorDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        input_embed_dim = embed_tokens.embedding_dim

        p_gen_input_size = input_embed_dim + self.output_embed_dim + 1
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        nn.init.zeros_(self.project_p_gens.bias)
        self.num_types = len(dictionary)
        self.num_embeddings = self.num_types

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = 0,
        alignment_heads: Optional[int] = 1,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        article_encoder_out = encoder_out[0]
        triples_encoder_out = encoder_out[1]
        align_encoder_out = [torch.cat((
            article_encoder_out['encoder_out'][0], 
            triples_encoder_out['encoder_out'][0]
        ), dim=0)]
        align_tokens = [torch.cat((
            article_encoder_out['src_tokens'][0], 
            triples_encoder_out['src_tokens'][0]
        ), dim=1)]
        encoder_padding_mask = [torch.cat((
            article_encoder_out['encoder_padding_mask'][0], 
            triples_encoder_out['encoder_padding_mask'][0]
        ), dim=1)]
        align_encoder_outs = {
            "encoder_out": align_encoder_out,
            "encoder_padding_mask":encoder_padding_mask,
            "src_tokens":align_tokens,
        }
        src_tokens = article_encoder_out['src_tokens'][0]
        src_entity_mask = article_encoder_out['src_entity_mask'][0]
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=align_encoder_outs,
            incremental_state=incremental_state,
        )
        if not features_only:
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            # project back to size of vocabulary
            logits = self.output_projection(x)
            gen_dists = self.get_normalized_probs_scriptable(
                (logits, None), log_probs=False, sample=None
            )
            batch_size = logits.shape[0]
            output_length = logits.shape[1]
            src_length = src_tokens.shape[1]
            assert logits.shape[2] == self.num_embeddings

            # attention of entity and triples
            attn = extra["attn"][0]
            # normalize the attention score
            triples_mask = torch.where(triples_encoder_out['encoder_padding_mask'][0]==True, 1, 0)
            triples_mask[:,-1] = 1
            attn_mask = torch.cat((src_entity_mask,triples_mask),dim=1)
            attn.masked_fill_(attn_mask[:,None,:].expand(attn.shape)==1,0)
            attn_sum = attn.sum(dim=-1)[:,:,None].expand(attn.shape)
            attn /= attn_sum
            # attn = F.softmax(attn.float().masked_fill(attn_mask[:,None,:].expand(attn.shape)==1,-1e9),dim=-1)

            # get the max entity probs in vocabulary distribution
            src_entity_mask = torch.where(
                src_entity_mask==0, torch.full_like(src_entity_mask, 1), torch.zeros_like(src_entity_mask)
            )
            index = src_tokens[:, None, :]
            src_entity_mask = src_entity_mask[:, None, :]
            index = index.expand(batch_size, output_length, src_length)
            src_entity_mask = src_entity_mask.expand(batch_size, output_length, src_length)
            src_entity_mask_dist_size = (batch_size, output_length, self.num_types)
            src_entity_mask_dist = src_entity_mask.new_zeros(src_entity_mask_dist_size)
            src_entity_mask_dist.scatter_add_(2, index, src_entity_mask)
            entity_probs_max = torch.where(
                src_entity_mask_dist>0, gen_dists, torch.full_like(gen_dists,0)
            ).max(dim=2)[0]
            entity_probs_sum = torch.where(
                src_entity_mask_dist>0, gen_dists, torch.full_like(gen_dists,0)
            ).sum(dim=2)
            extra["entity_probs_max"] = entity_probs_max
            
            # the copy prob: p_gens
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, x, entity_probs_sum[:,:,None]), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens.float())
            assert article_encoder_out is not None
            assert attn is not None

            x = self.output_layer(gen_dists, attn, align_tokens[0], p_gens)
        return x, extra

    def output_layer(
        self,
        gen_dists: Tensor,
        attn: Tensor,
        src_tokens: Tensor,
        p_gens: Tensor
    ) -> Tensor:
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """
        assert gen_dists.shape[2] == self.num_types
        batch_size = gen_dists.shape[0]
        output_length = gen_dists.shape[1]
        src_length = src_tokens.shape[1]
        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        # p_gens = torch.where(torch.isnan(attn[:,:,:1]), torch.full_like(p_gens, 1), p_gens)
        # attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn) # NaN
        gen_dists = torch.mul(gen_dists, p_gens)
        attn = torch.mul(attn.float(), 1 - p_gens)

        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn.float())

        # Final distributions, [batch_size, output_length, num_types].
        return gen_dists + attn_dists

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs

    # The following codes are used when evaluating the relation consistency
    def incremental_forward(
        self,
        next_token_probs,
        token_len,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        article_encoder_out = encoder_out[0]
        triples_encoder_out = encoder_out[1]
        align_encoder_out = [torch.cat((
            article_encoder_out['encoder_out'][0], 
            triples_encoder_out['encoder_out'][0]
        ), dim=0)]
        align_tokens = [torch.cat((
            article_encoder_out['src_tokens'][0], 
            triples_encoder_out['src_tokens'][0]
        ), dim=1)]
        encoder_padding_mask = [torch.cat((
            article_encoder_out['encoder_padding_mask'][0], 
            triples_encoder_out['encoder_padding_mask'][0]
        ), dim=1)]
        align_encoder_outs = {
            "encoder_out": align_encoder_out,
            "encoder_padding_mask":encoder_padding_mask,
            "src_tokens":align_tokens,
        }
        src_entity_mask = article_encoder_out['src_entity_mask'][0]
        x, extra = self.incremental_extract_features(
            next_token_probs,
            token_len,
            encoder_out=align_encoder_outs,
            incremental_state=incremental_state,
        )

        logits = self.output_projection(x)
        src_tokens = article_encoder_out['src_tokens'][0]
        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        src_length = src_tokens.shape[1]
        assert logits.shape[2] == self.num_embeddings
        
        attn: Optional[Tensor] = extra["attn"][0]
        triples_mask = torch.where(triples_encoder_out['encoder_padding_mask'][0]==True, 1, 0)
        triples_mask[:,-1] = 1
        attn_mask = torch.cat((src_entity_mask,triples_mask),dim=1)
        attn.masked_fill_(attn_mask[:,None,:].expand(attn.shape)==1,0)
        attn_sum = attn.sum(dim=-1)[:,:,None].expand(attn.shape)
        attn /= attn_sum

        gen_dists = self.get_normalized_probs_scriptable(
            (logits, None), log_probs=False, sample=None
        )

        assert logits.shape[2] == self.num_embeddings
        if len(next_token_probs.size()) == 2:
            next_token_embed = self.embed_tokens(next_token_probs)
        else:
            next_token_embed = torch.matmul(next_token_probs.to(self.embed_tokens.weight), self.embed_tokens.weight)
        
        # the max entity probs in vocabulary distribution
        src_entity_mask = torch.where(
            src_entity_mask==0, torch.full_like(src_entity_mask, 1), torch.zeros_like(src_entity_mask)
        )

        index = src_tokens[:, None, :]
        src_entity_mask = src_entity_mask[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        src_entity_mask = src_entity_mask.expand(batch_size, output_length, src_length)
        src_entity_mask_dist_size = (batch_size, output_length, self.num_types)
        src_entity_mask_dist = src_entity_mask.new_zeros(src_entity_mask_dist_size)
        src_entity_mask_dist.scatter_add_(2, index, src_entity_mask)
        entity_probs_sum = torch.where(
            src_entity_mask_dist>0, gen_dists, torch.full_like(gen_dists,0)
        ).sum(dim=2).to(x)
        
        next_token_embed *= self.embed_scale
        predictors = torch.cat((next_token_embed, x, entity_probs_sum[:,:,None]), 2)
        p_gens = self.project_p_gens(predictors)
        p_gens = torch.sigmoid(p_gens.float())

        assert attn is not None

        x = self.output_layer(gen_dists, attn, align_tokens[0], p_gens)
        return x, extra

    def incremental_extract_features(
        self,
        next_token_probs,
        token_len,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        bs = next_token_probs.size()[0]
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None

        assert incremental_state is not None
        assert self.embed_positions is not None
        positions = self.embed_positions(
            torch.zeros([bs, token_len]).to(self.embed_positions.weight.device).int(), 
            incremental_state=incremental_state
        )
        positions = positions[:, -1:]
        # embed tokens and positions
        if len(next_token_probs.size()) == 2:
            x = self.embed_scale * self.embed_tokens(next_token_probs)
        else:
            x = self.embed_scale * torch.matmul(next_token_probs.to(self.embed_tokens.weight), self.embed_tokens.weight)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
