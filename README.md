# legal-rag-eval

[![CI](https://github.com/m3metic/legal-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/m3metic/legal-rag-eval/actions/workflows/ci.yml)

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

## Continuous evaluation

Retrieval quality is tested on every push and pull request, the same way code is. A GitHub Actions workflow runs lint and unit tests, then an **evaluation gate**: it reruns BM25 on both full benchmarks (5,622 queries, about 3 minutes) and compares each metric with the baseline in [`configs/gate.yaml`](configs/gate.yaml). If any metric falls by more than 0.005, the run fails and the pull request is blocked. The comparison table appears on the run's summary page.

Each metric is checked on each dataset separately, because a change can help one and hurt another. Removing the stopword filter from the BM25 tokenizer, for example, raised MAUD recall@1 (+0.010) but lowered ObliQA recall@10 (-0.011) and recall@1000w (-0.024). An average would have hidden that; the gate failed 5 of its 8 checks.

When a change is meant to improve the numbers, `python -m legalrag gate --update` writes the new scores into the gate file. Committing it makes the improvement the new floor.

## Answer generation with citations

`python -m legalrag answer configs/answers/obliqa_bm25.yaml` takes a fixed random sample of 100 ObliQA test questions. For each one, a writer model answers from the top 5 retrieved passages and cites them as `[1]`, `[2]`. If the passages don't answer the question, it says so. ObliQA has no reference answers, so the passages labelled as answering each question serve as the reference. The answers are scored in two ways:

- **Citation scores, with no model involved:**
  - **Citation precision:** how many cited passages are among the correct ones.
  - **Citation recall:** how many correct passages were cited.
  - **Citation recall within the context:** of the correct passages the writer was shown, how many it cited. This separates "retrieval missed it" from "the writer ignored it".
  - **Rates** of answers citing passages they weren't shown, answers with no citations, and answers declining to answer.
- **Grader scores:** a second model splits each answer into claims and checks each one against the passages the answer cites (**faithfulness**). It also lists the obligations in the correct passages and checks which ones the answer states (**obligation coverage**), and flags any contradictions.

The writer (Claude Haiku) and the grader (an OpenAI model) come from different providers on purpose. AI graders tend to rate answers from their own model family more kindly, and the results file flags a run where they share a family. Models are set in the config as `provider:model`, so either can be swapped. Every model response is cached in `.cache/llm/`, so rerunning only pays for calls that changed.

Because this calls paid APIs, it runs weekly and on demand in GitHub Actions ([`answers.yml`](.github/workflows/answers.yml)), not on every push. It needs the `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` repository secrets. Without the grader's key, it still reports the citation scores.

## Quickstart

```bash
git clone https://github.com/m3metic/legal-rag-eval && cd legal-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"
bash scripts/download_data.sh                                  # both datasets, ~130 MB

python -m pytest                                               # 31 tests
python -m legalrag run configs/obliqa/bm25_passage.yaml        # one experiment
python -m legalrag compare results/obliqa_*                    # markdown results table
python -m legalrag inspect results/maud_hybrid_section300 \
    --group "No-Shop" --n 5                                    # read failures side by side
python -m legalrag gate                                        # fail if quality drops vs baseline

pip install -e ".[llm]"                                        # answer generation (needs API keys)
python -m legalrag answer configs/answers/obliqa_bm25.yaml     # cited answers, graded
```

Models: `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranking), downloaded automatically on first use. Embeddings are cached in `.cache/`.

## Project layout

```
.github/workflows/    ci.yml: lint, tests and the evaluation gate on every push and pull request;
                      answers.yml: weekly answer-quality run
configs/              one YAML file per experiment; maud_queries.yaml holds the MAUD search queries;
                      gate.yaml holds the baseline scores the CI gate enforces;
                      answers/ holds answer-generation configs
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
  gate.py             evaluation gate: rerun experiments, compare with baseline, pass or fail
  llm.py              one interface over Anthropic and OpenAI models, with a response cache
  answering.py        cited answers, citation scores and grading
tests/                metrics, chunk offsets, ground-truth location, retrievers, gate logic,
                      answering (with stand-in models, no API calls)
```

## Limitations and future work

- ObliQA's questions were generated by an LLM and then validated, not written from scratch by compliance officers.
- MAUD covers US public-company deals from 2021 only; 34 of 2,870 excerpts (1.2%) could not be located and are excluded.
- One MAUD query was tuned after reading test-set failures; the gain was confirmed on held-out contracts.
- Answer quality is graded by a model, which is cheaper than lawyers but not a substitute: a sample of grades should be checked by hand before the grader's scores are trusted. Answer generation covers ObliQA only so far.
- Next steps: answer-quality baselines added to the gate once the first real run is in, end-to-end accuracy on MAUD's multiple-choice deal-point questions, fine-tuning the embedding model on ObliQA's training split, and query rewrites for the remaining weak clause types.

## Data and citations

Datasets are downloaded from their original sources and are not redistributed here.

- ObliQA: Gokhan, Wang, Gurevych & Briscoe. *RIRAG: Regulatory Information Retrieval and Answer Generation.* 2024. [arXiv:2409.05677](https://arxiv.org/abs/2409.05677)
- MAUD: Wang et al. *MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding.* EMNLP 2023. [arXiv:2301.00876](https://arxiv.org/abs/2301.00876). CC BY 4.0.

Code in this repository is released under the [MIT License](LICENSE).
