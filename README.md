# Faithful Abstractive Summarization via Fact-aware Consistency-constrained Transformer(CIKM 2022)

## Abstract
Abstractive summarization is a classic task in Natural Language Generation (NLG), which aims to produce a concise summary of the original document. Recently, great efforts have been made on sequence-to-sequence neural networks to generate abstractive summaries with a high level of fluency. However, prior arts mainly focus on the optimization of token-level likelihood, while the rich semantic information in documents has been largely ignored. In this way, the summarization results could be vulnerable to hallucinations, i.e., the semantic-level inconsistency between a summary and corresponding original document. To deal with this challenge, in this paper, we propose a novel fact-aware abstractive summarization model, named Entity-Relation Pointer Generator Network (ERPGN). Specially, we attempt to formalize the facts in original document as a factual knowledge graph, and then generate the high-quality summary via directly modeling consistency between summary and the factual knowledge graph. To that end, we first leverage two pointer network structures to capture the fact in original documents. Then, to enhance the traditional token-level likelihood loss, we design two extra semantic-level losses to measure the disagreement between a summary and facts from its original document. Extensive experiments on public datasets demonstrate that our ERPGN framework could outperform both classic abstractive summarization models and the state-of-the-art fact-aware baseline methods, with significant improvement in terms of faithfulness.


## How to Run it
```bash
pip install fairseq

# modify train.sh, test.sh according to your requirements

bash train.sh # train your model

bash test.sh # test your model
```

## Preprocess Datasets and Models

[**Models**](https://drive.google.com/drive/u/0/folders/1gdXsWamuYtOHZhMVZVHwPviIeR--r8yr): bart_rc.pt (including the BART pre-trained model and a Discriminator for Relation Consistency Identification)

[**Datasets**](https://drive.google.com/drive/u/0/folders/1gdXsWamuYtOHZhMVZVHwPviIeR--r8yr): Xsum

