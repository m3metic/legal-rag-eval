# legal-rag-eval

**How well does retrieval work on real legal documents, and what actually improves it?**

A retrieval-evaluation framework for legal RAG, tested on two expert-labelled benchmarks:

| Use case | Dataset | Corpus | Questions | Ground truth |
|---|---|---|---|---|
| **Regulatory compliance** | [ObliQA](https://github.com/RegNLP/ObliQADataset) | 40 Abu Dhabi Global Market (ADGM) financial regulations, 13k passages | 2,786 test questions a compliance officer would ask | Passage IDs that answer each question |
| **M&A due diligence** | [MAUD](https://github.com/TheAtticusProject/maud) | 152 public merger agreements, ~52k words each | 2,836 (agreement × deal point) pairs, 22 deal points | Clause spans marked by lawyers |

Most RAG demos never measure whether the right text was retrieved. In a compliance or diligence setting, a missed obligation or a missed termination-fee clause is the failure that matters, so this project starts from measurement and treats every design choice (chunking, retriever, query wording) as an experiment.

> 🚧 Work in progress. BM25 baselines are done; dense, hybrid and reranking runs are next.

## Results so far

**ObliQA (compliance), BM25.** Our passage-level baseline gets recall@10 of 0.754, close to the 0.761 reported by the dataset authors ([Gokhan et al., 2024](https://arxiv.org/abs/2409.05677)), which is a useful check that the evaluation code is correct.

| experiment | chunks | words/chunk | recall@1 | recall@5 | recall@10 | MRR | recall@500w | recall@1000w |
|---|---|---|---|---|---|---|---|---|
| bm25 · passage | 13,012 | 64 | 0.514 | 0.694 | 0.754 | 0.673 | **0.716** | **0.765** |
| bm25 · passage + section headers | 13,012 | 73 | 0.513 | 0.695 | 0.747 | 0.671 | 0.700 | 0.758 |
| bm25 · 3-passage windows | 6,495 | 194 | **0.541** | **0.735** | **0.794** | **0.691** | 0.614 | 0.716 |

**MAUD (M&A), BM25, searching within the agreement under review.**

| experiment | words/chunk | recall@1 | recall@5 | recall@10 | hit@5 | MRR | recall@1000w |
|---|---|---|---|---|---|---|---|
| bm25 · 200-word windows | 200 | 0.323 | 0.594 | 0.688 | 0.792 | 0.605 | **0.594** |
| bm25 · section-aware chunks | 219 | **0.341** | **0.631** | **0.721** | **0.813** | **0.620** | 0.553 |
| bm25 · section-aware, query = bare deal-point name | 219 | 0.117 | 0.268 | 0.367 | 0.327 | 0.231 | 0.224 |

### Early findings

1. **"Bigger chunks win" is partly a measurement artefact.** Three-passage windows beat single passages on recall@10 (0.794 vs 0.754), but they send three times as much text to the LLM. At a fixed context budget of 1,000 words, single passages win (0.765 vs 0.716). That's why every experiment reports budget-based recall alongside recall@k.
2. **Finding *all* the obligations is the hard part.** For ObliQA questions backed by one passage, recall@10 is 0.83. For questions backed by 3 passages it drops to 0.39, even though at least one relevant passage is still in the top 10 for 88% of those questions. A compliance answer built on a partial set of obligations is a wrong answer.
3. **Query wording matters more than any retriever tweak so far.** On MAUD, searching for "MAE Definition" instead of a written-out question cuts recall@5 from 0.63 to 0.27.
4. **Some deal points are much harder than others.** Recall@5 ranges from 0.97 (specific performance) to 0.21 (negative interim operating covenants), which are long lists of restrictions spread across a section.

## How it works

**Gold units, not chunks.** Each dataset defines its smallest unit of evidence: an ObliQA passage ID, or a character span in a MAUD agreement. Every chunk records which units it covers, so any chunking strategy can be scored against the same ground truth.

**Making MAUD a retrieval benchmark.** MAUD was built for clause classification and stores each excerpt as pieces separated by `<omitted>`. The loader locates each piece in the full agreement (normalising punctuation and whitespace, with anchor matching when annotators trimmed words), recovering spans for 2,836 of 2,870 annotated items (98.8%).

**Metrics.** recall@k, hit@k, MRR, nDCG@k, and recall at a fixed word budget (500 / 1,000 / 2,000 words). A MAUD span counts as found when a chunk overlaps it by at least 200 characters (or half the span, if shorter).

**Retrievers.** BM25, dense embeddings (any sentence-transformers model, with instruction prefixes), hybrid via reciprocal rank fusion, and cross-encoder reranking on top of any of them. Embeddings are cached on disk by model and corpus hash.

## Quickstart

```bash
git clone https://github.com/<you>/legal-rag-eval && cd legal-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"
./scripts/download_data.sh          # ~130 MB, both datasets

python -m legalrag run configs/obliqa/bm25_passage.yaml --limit 200   # quick check
python -m legalrag run configs/obliqa/*.yaml                          # full ObliQA suite
python -m legalrag compare results/obliqa_*                            # markdown table
pytest
```

Each run writes `results/<name>/metrics.json` (scores, per-group breakdowns, timings) and `per_query.jsonl` (every query's ranked results, for error analysis).

## Project layout

```
configs/            one YAML per experiment; maud_queries.yaml holds the MAUD search queries
scripts/            dataset download
src/legalrag/
  types.py          Chunk, EvalItem, Span
  metrics.py        rank-based and budget-based retrieval metrics
  chunking.py       word windows and section-aware chunking for contracts
  retrievers.py     BM25, dense, hybrid (RRF), reranking
  datasets/         ObliQA and MAUD loaders
  experiment.py     run a config, save results, compare runs
tests/              metrics, chunk offsets, MAUD span location, retrievers
```

## Data and citations

Datasets are downloaded from their original sources and not redistributed here.

- ObliQA: Gokhan, Wang, Gurevych & Briscoe. *RIRAG: Regulatory Information Retrieval and Answer Generation.* 2024. [arXiv:2409.05677](https://arxiv.org/abs/2409.05677)
- MAUD: Wang et al. *MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding.* EMNLP 2023. [arXiv:2301.00876](https://arxiv.org/abs/2301.00876). CC BY 4.0.
