import torch
import torch.nn as nn
import torch.nn.functional as F

# Relation Consistency Discriminator
class RCDiscriminator(nn.Module):
    def __init__(self, num_embeddings, d_model, num_classes=3):
        # class: 0 consistent, 1: inconsistent, 2: irrelevant
        super().__init__()
        self.summarization_emb= nn.Embedding(num_embeddings, d_model)
        self.triples_emb = nn.Embedding(num_embeddings, d_model)

        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        nn.init.normal_(self.classifier.weight, mean=0, std=d_model ** -0.5)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, summaries, src_triples):
        # delimiter between triples(relations) and summaries
        start_token = torch.full_like(src_triples, 2)[:,:1]
        # truncate triples
        if src_triples.shape[1] > 900:
            src_triples = src_triples[:, :900]
            src_triples = torch.cat((src_triples,start_token), 1)
        
        src_triples = torch.cat((start_token,src_triples), 1)

        if len(summaries.size()) == 2:
            # summaries(batch_size, seq_len)
            summaries_emb = self.summarization_emb(summaries)
        else:
            # summary probabilities(batch_size, seq_len, len_dict)
            summaries_emb = torch.matmul(summaries, self.summarization_emb.weight)
        
        triples_emb = self.triples_emb(src_triples)
        enc_input = torch.cat((summaries_emb, triples_emb), 1)
        
        x, _ = self.lstm(enc_input)
        # dimension (batch_size, seq_len, hidden_size)
        x = x[:, -1, :] 
        x = self.dropout(x)
        logits = self.classifier(x)

        return F.log_softmax(logits, -1)
