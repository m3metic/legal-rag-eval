# legal-rag-eval

**How well does retrieval work on real legal documents, and what actually improves it?**

An evaluation framework for the retrieval step of legal RAG systems, tested on two expert-labelled benchmarks:

| Use case | Dataset | Corpus | Questions | Ground truth |
|---|---|---|---|---|
| Regulatory compliance | [ObliQA](https://github.com/RegNLP/ObliQADataset) | 40 Abu Dhabi Global Market (ADGM) financial regulations, 13k passages | 2,786 test questions | Passages that answer each question |
| M&A due diligence | [MAUD](https://github.com/TheAtticusProject/maud) | 152 public merger agreements, ~52k words each | 2,836 contract × clause-type pairs (22 clause types) | Clause text marked by lawyers |

In compliance and due diligence, a missed obligation or a missed termination clause is the failure that matters. So this project measures whether the right text is retrieved, and treats every design choice (chunking, retriever, query wording) as an experiment.

## Results

**Compliance (ObliQA, 2,786 questions)**

| Retriever · chunking | recall@1 | recall@10 | MRR | recall@1000w |
|---|---|---|---|---|
| BM25 · single passage | 0.514 | 0.754 | 0.673 | 0.765 |
| BM25 · 3-passage windows | 0.541 | **0.794** | 0.691 | 0.716 |
| Dense (bge-small) · single passage | 0.435 | 0.686 | 0.582 | 0.688 |
| Hybrid (BM25 + bge-small) · single passage | 0.513 | 0.763 | 0.673 | **0.772** |
| Hybrid + reranker (20 candidates) · passage + section titles | **0.570** | 0.771 | **0.719** | 0.770 |

The BM25 baseline (recall@10 0.754) reproduces the dataset authors' published figure of 0.761 to within a point.

**M&A due diligence (MAUD, 2,836 items, searching within the agreement under review)**

| Retriever · chunking | recall@1 | recall@5 | recall@10 | MRR | recall@1000w |
|---|---|---|---|---|---|
| BM25 · 200-word windows | 0.339 | 0.632 | 0.739 | 0.627 | 0.632 |
| BM25 · section-aware chunks | 0.366 | 0.693 | 0.792 | 0.654 | 0.609 |
| BM25 · section-aware, query = bare clause name | 0.118 | 0.281 | 0.379 | 0.237 | 0.226 |
| Dense (bge-small) · section-aware | 0.351 | 0.648 | 0.752 | 0.607 | 0.570 |
| Hybrid (BM25 + bge-small) · section-aware | **0.399** | **0.721** | **0.828** | **0.685** | **0.635** |

*recall@1000w* = share of the correct evidence that fits in the first 1,000 words of results, roughly what an LLM's context budget allows. See [docs/methodology.md](docs/methodology.md) for all metric definitions.

## Key findings

1. **Bigger chunks look better until you count the cost.** 3-passage windows beat single passages on recall@10 (0.794 vs 0.754), but at a fixed 1,000-word context budget single passages win (0.765 vs 0.716). Reporting budget-based recall reversed the conclusion.
2. **Query wording mattered more than the retriever.** Searching for a bare clause name ("MAE Definition") instead of a written question cut recall@10 from 0.79 to 0.38. Rewriting one MAUD query in the contracts' own vocabulary raised retrieval of negative interim operating covenants from 0.24 to 0.76 recall@5, and the gain held on 146 contracts that were never inspected.
3. **Embeddings help more on contracts than on regulations.** A small embedding model lost to BM25 on both datasets, but hybrid search added +3.6 points recall@10 on contracts against +0.9 on regulations. A likely reason: ObliQA's questions were generated from the passages that answer them, so they reuse the passage wording, which favours keyword search.
4. **Reranking improves the top of the list, at a price.** A cross-encoder reranker raised recall@1 from 0.514 to 0.570 on ObliQA. Re-scoring 20 candidates instead of 50 kept the gains at about a third of the cost. The full pipeline (recall@10 0.771) comes within about a point of the 0.783 reported for a 7-billion-parameter embedding model, using models that run on a laptop.
5. **Finding all the obligations is the hard part.** For ObliQA questions backed by one passage, recall@10 is 0.83; for questions backed by three, it falls to 0.39, even though at least one correct passage is retrieved for 88% of them.
6. **Reading failures found three bugs the averages did not:** a metric that could exceed its maximum, a chunker that treated cross-references as section headings, and ground truth that pointed at the table of contents. Details in the [methodology](docs/methodology.md#bugs-found-through-error-analysis).

## How it works

- **Ground truth is defined once, independent of chunking.** Each dataset defines its smallest unit of evidence (an ObliQA passage, or a character span in a MAUD contract). Every chunk records what it covers, so any chunking strategy is scored against the same answer key.
- **MAUD converted to a retrieval benchmark.** MAUD was built for clause classification. Each lawyer-marked excerpt is located in the full contract text, with matches chosen to keep an excerpt's pieces together; 98.8% of items are recovered.
- **Retrievers:** BM25, dense embeddings (any sentence-transformers model), hybrid via reciprocal rank fusion, and cross-encoder reranking on top of any of them.
- **Every experiment is a YAML config.** Results record their own config, per-group breakdowns and timings, plus a per-question log for error analysis.

## Quickstart

```bash
git clone https://github.com/m3metic/legal-rag-eval && cd legal-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"
bash scripts/download_data.sh                                  # both datasets, ~130 MB

python -m pytest                                               # 15 tests
python -m legalrag run configs/obliqa/bm25_passage.yaml        # one experiment
python -m legalrag compare results/obliqa_*                    # markdown results table
python -m legalrag inspect results/maud_hybrid_section300 \
    --group "No-Shop" --n 5                                    # read failures side by side
```

Models: `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranking), downloaded automatically on first use. Embeddings are cached in `.cache/`.

## Project layout

```
configs/              one YAML file per experiment; maud_queries.yaml holds the MAUD search queries
docs/methodology.md   full write-up: datasets, ground truth, metrics, experiments, error analysis
results/              metrics.json for every run reported above
scripts/              dataset download
src/legalrag/
  datasets/           ObliQA and MAUD loaders
  chunking.py         word windows and section-aware chunking
  retrievers.py       BM25, dense, hybrid (RRF), reranking
  metrics.py          rank-based and budget-based retrieval metrics
  experiment.py       run a config, save results, compare runs
  inspect_failures.py side-by-side view of failed queries
tests/                metrics, chunk offsets, ground-truth location, retrievers
```

## Limitations and future work

- ObliQA's questions were generated by an LLM and then validated, not written from scratch by compliance officers.
- MAUD covers US public-company deals from 2021 only; 34 of 2,870 excerpts (1.2%) could not be located and are excluded.
- One MAUD query was tuned after reading test-set failures; the gain was confirmed on held-out contracts.
- This measures retrieval only. Next steps: answer generation with citations, end-to-end accuracy on MAUD's multiple-choice deal-point questions, fine-tuning the embedding model on ObliQA's training split, and query rewrites for the remaining weak clause types.

## Data and citations

Datasets are downloaded from their original sources and are not redistributed here.

- ObliQA: Gokhan, Wang, Gurevych & Briscoe. *RIRAG: Regulatory Information Retrieval and Answer Generation.* 2024. [arXiv:2409.05677](https://arxiv.org/abs/2409.05677)
- MAUD: Wang et al. *MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding.* EMNLP 2023. [arXiv:2301.00876](https://arxiv.org/abs/2301.00876). CC BY 4.0.

Code in this repository is released under the [MIT License](LICENSE).
