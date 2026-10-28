# Methodology

This document describes how the evaluation was built and run: the datasets, how ground truth was constructed, the metrics, the experiments, and what error analysis found. The [README](../README.md) has the headline results.

## 1. Problem and scope

Retrieval-augmented generation (RAG) answers questions by first retrieving relevant passages from a document collection and then generating an answer from them. If the right text is never retrieved, no prompt can fix the answer. This project therefore evaluates the **retrieval** step on its own, where expert-labelled ground truth makes objective measurement possible.

Two legal use cases stress retrieval differently:

| Use case | Task | Difficulty |
|---|---|---|
| Regulatory compliance | Find every rule that applies to a question, across 40 regulations | Answers often need several passages from different places |
| M&A due diligence | Find a specific clause inside one ~52,000-word merger agreement | Long, repetitive documents full of defined terms and cross-references |

## 2. Datasets

### Selection criteria

Ground truth linking each question to the exact evidence was the main requirement. Others: expert labels, full-length real documents, realistic questions, enough questions to detect small differences, a clear licence, and a published paper with baseline numbers.

| Option considered | Outcome |
|---|---|
| Hand-written questions over chosen documents | Too small and not expert-validated |
| Merger agreements scraped from SEC EDGAR, or EU regulations | Realistic documents but no answer key |
| CUAD | Expert clause labels, but commercial contracts rather than M&A |
| **MAUD** | Chosen for M&A |
| **ObliQA** | Chosen for compliance |

### ObliQA

- 40 ADGM financial regulation documents, pre-split into 13,732 numbered passages (720 empty structural markers are skipped). Median passage length 40 words.
- Questions are phrased as a compliance officer asking the regulator for clarification. Each lists the passages that answer it.
- The 2,786-question **test split** is used for every result. The 22,295-question training split is reserved for fine-tuning and never used for scoring.
- Test questions by number of answering passages: 2,126 need one, 506 need two, 105 need three, 49 need four or more.

### MAUD

- 152 public merger agreements (median about 52,000 words) annotated by lawyers for the deal points of the 2021 ABA Public Target Deal Points Study. Licensed CC BY 4.0.
- MAUD's 92 questions share 22 underlying clause types. For retrieval, items are de-duplicated to one per contract and clause type: 2,870 items.
- Only the full "main" excerpts are used; "abridged" rows are shortened duplicates.
- Search is restricted to the agreement under review, as a reviewer working through a data room would.

## 3. Ground truth

### Gold units

The answer key is defined at the smallest level each dataset labels, called a **gold unit**:

- ObliQA: a passage, keyed as `document::passage_id`.
- MAUD: a character span in the contract.

Every chunk records which gold units it contains (ObliQA) or which character range it covers (MAUD). Any chunking strategy is scored against the same answer key, so chunking can change without touching the ground truth.

### Locating MAUD excerpts

MAUD provides excerpt text but not positions, and each excerpt is several pieces joined by an `<omitted>` marker. Locating them took several iterations:

| Method | Result |
|---|---|
| Exact search for the whole excerpt | 1 of 2,870 found |
| Split on `<omitted>`, normalise each piece (lowercase, letters and digits only), exact search | 54% of pieces found |
| Anchor matching: first 10 words, then last 10 words after that point | 2,836 of 2,870 items located (98.8%) |
| Choose among multiple matches (see below) | Same coverage, correct positions |

A normalisation map records which original character each normalised character came from, so matches convert back to exact character offsets.

27% of pieces (2,495 of 9,148) match more than one place, usually because a section heading also appears in the table of contents, or because boilerplate repeats. The loader collects every match and picks the one closest to the excerpt's unambiguous pieces; with no unambiguous pieces, it picks the one closest to the median of each piece's last match, which steers away from the table of contents. The 34 excerpts that could not be located are excluded and reported.

### Matching rule for spans

A chunk finds a MAUD gold span if it overlaps it by at least 200 characters, or by half the span if the span is shorter than 400 characters, so that a chunk merely grazing a clause does not count.

## 4. Metrics

All metrics are computed per question and averaged.

| Metric | Meaning |
|---|---|
| recall@k | Share of all gold evidence that appears in the top k results |
| hit@k | Whether at least one piece of gold evidence is in the top k |
| MRR | 1 / rank of the first correct result |
| nDCG@k | Rank-weighted recall; a chunk earns credit only for evidence not already found higher up, so overlapping chunks cannot push it above 1 |
| recall@Nw | Fill a budget of N words (500 / 1,000 / 2,000) with results in rank order, counting only chunks that fit whole; share of gold evidence inside |

recall@k favours large chunks, since ten 200-word chunks hold more text than ten 60-word chunks. The real constraint is how much text an LLM can be given, so **recall@Nw** is the fair comparison between chunking strategies.

## 5. Chunking

| Dataset | Strategy | Description |
|---|---|---|
| ObliQA | `passage` | One chunk per non-empty passage (the dataset's native unit) |
| ObliQA | `passage_header` | Passage prefixed with its parent section titles |
| ObliQA | `window` | 3 neighbouring passages, sliding by 2 |
| MAUD | `window` | 200-word windows with 50-word overlap |
| MAUD | `section` | Split at contract headings (`Section 7.1 Termination.`, `ARTICLE VI`), window long sections at 300 words, prefix each chunk with its heading |

Every MAUD chunk keeps the character offsets of its source text, which a unit test checks.

## 6. Retrievers

| Retriever | How it works |
|---|---|
| BM25 | Keyword scoring (`rank_bm25`, k1 = 1.5, b = 0.75, small stopword list) |
| Dense | `BAAI/bge-small-en-v1.5` embeddings with its recommended query instruction; cosine similarity |
| Hybrid | Reciprocal rank fusion of BM25 and dense: each result scores the sum of 1 / (60 + rank). Uses ranks only, so no score calibration is needed |
| Rerank | Top 20 or 50 hybrid candidates re-scored by the cross-encoder `BAAI/bge-reranker-base` |

All models are free and run locally. Embeddings are cached on disk, keyed by a hash of the model name and every chunk's text.

## 7. Experiments and results

All runs are defined by files in `configs/`, and each writes `results/<name>/metrics.json`. Timings are from a MacBook Air (Apple silicon).

### ObliQA (2,786 questions)

| Retriever · chunking | recall@1 | recall@10 | MRR | nDCG@10 | recall@1000w |
|---|---|---|---|---|---|
| BM25 · passage | 0.514 | 0.754 | 0.673 | 0.654 | 0.765 |
| BM25 · passage + section titles | 0.513 | 0.747 | 0.671 | 0.651 | 0.758 |
| BM25 · 3-passage windows | 0.541 | 0.794 | 0.691 | 0.672 | 0.716 |
| Dense · passage | 0.435 | 0.686 | 0.582 | 0.576 | 0.688 |
| Hybrid · passage | 0.513 | 0.763 | 0.673 | 0.659 | 0.772 |
| Hybrid + rerank (20) · passage + section titles | 0.570 | 0.771 | 0.719 | 0.694 | 0.770 |

For comparison, Gokhan et al. (2024) report recall@10 of 0.761 for BM25 and 0.783 for NV-Embed-v2. Setups differ in detail, so treat that comparison as approximate.

**Reranker cost** (first 300 questions):

| Setup | recall@1 | recall@10 | MRR | ms per question |
|---|---|---|---|---|
| Hybrid + section titles | 0.491 | 0.743 | 0.645 | 68 |
| + reranker, 50 candidates | 0.547 | 0.752 | 0.694 | 3,162 |
| + reranker, 20 candidates | 0.552 | 0.741 | 0.696 | 1,166 |

### MAUD (2,836 items)

| Retriever · chunking | recall@1 | recall@5 | recall@10 | hit@10 | MRR | recall@1000w |
|---|---|---|---|---|---|---|
| BM25 · 200-word windows | 0.339 | 0.632 | 0.739 | 0.913 | 0.627 | 0.632 |
| BM25 · section-aware | 0.366 | 0.693 | 0.792 | 0.936 | 0.654 | 0.609 |
| BM25 · section-aware, bare clause name as query | 0.118 | 0.281 | 0.379 | 0.458 | 0.237 | 0.226 |
| Dense · section-aware | 0.351 | 0.648 | 0.752 | 0.885 | 0.607 | 0.570 |
| Hybrid · section-aware | 0.399 | 0.721 | 0.828 | 0.949 | 0.685 | 0.635 |

## 8. Findings

1. **Chunk size and budget.** 3-passage windows win on recall@10 but lose at a fixed 1,000-word budget (0.716 vs 0.765). The same reversal appears on MAUD (section-aware chunks win on recall@10, 200-word windows on recall@1000w).
2. **Multi-passage questions.** ObliQA recall@10 is 0.83 for one-passage questions and 0.39 for three-passage questions, while hit@10 stays around 0.84–0.88. Retrieval usually finds one relevant rule but rarely all of them.
3. **Dense vs keyword by dataset.** Relative to BM25 recall@10, dense alone is −6.8 points on ObliQA and −4.0 on MAUD; hybrid is +0.9 and +3.6. Hypothesis: ObliQA's LLM-generated questions reuse the wording of their source passages, favouring keyword search. Measuring question–passage word overlap would test this.
4. **Reranking** adds +5.6 points recall@1 on ObliQA. 20 candidates keep the top-of-list gains of 50 at 37% of the time; recall@10 changes little because a reranker can only reorder the candidates it receives.
5. **Clause types vary widely.** MAUD BM25 recall@5 ranges from 0.99 (specific performance) to 0.32 (breach of meeting covenant). The hardest types are termination rights and remedies for breach, whose text is spread across sections, and the representations-and-warranties closing condition, whose gold evidence averages 15 separate spans per contract.
6. **Query wording.** See section 9.

## 9. Error analysis

`python -m legalrag inspect <run> --group <clause type>` prints failed queries with the gold text and the retrieved text side by side.

**Negative interim operating covenants** (initially recall@5 0.24 with BM25). In five sampled failures:

- The query ("Actions the company may not take before closing without parent's consent") shared almost no vocabulary with the clauses, which say "conduct of business pending the merger", "the Company shall not, and shall not permit any of its Subsidiaries to", "forbearance" and "interim period".
- The query's one distinctive shared word, "consent", matched unrelated clauses that also require Parent's consent (litigation settlement, assignment, appraisal demands).
- The mirror-image covenant restricting *Parent* was retrieved, since it uses the same vocabulary.

Rewriting the query in the contracts' vocabulary ("Conduct of business pending the merger: the Company shall not, and shall not permit its Subsidiaries to, take the following actions without the prior written consent of Parent during the interim period") raised BM25 recall@5 from 0.242 to 0.757 and recall@10 from 0.394 to 0.860. Because the wording was chosen from test-set failures, it was checked on the 146 contracts that were not inspected: hybrid recall@5 rose from 0.202 to 0.759 there, matching the full set. (The held-out check was run before the ground-truth fix in section 10; the BM25 figures above use the corrected ground truth.)

**Representations-and-warranties closing condition.** Most "failures" retrieved the condition itself in the top three. Low recall comes from the ground truth, which also marks every representation the condition refers to (about 15 spans per contract). hit@5 (0.80) is the more meaningful measure for this clause type. One genuine miss was a tender-offer agreement, where the clause is titled "Conditions to the Offer" and phrased negatively ("shall not be true and correct").

## 10. Bugs found through error analysis

None of these showed up in the aggregate metrics; each was found by reading individual results.

| Bug | How it was found | Fix | Effect |
|---|---|---|---|
| nDCG could exceed 1 when overlapping windows contained the same gold passage | A results table showed nDCG@10 of 0.868 for windows, far above other runs | Credit a chunk only for evidence not already found higher in the ranking | Windows nDCG@10 corrected to 0.672; other metrics unaffected |
| The section chunker treated cross-references in running text ("set forth in Article VII") as headings, about 3,900 times across 152 contracts | Retrieved chunks from Section 5.2 were labelled `[Article VII]` | Ignore heading matches that follow a lowercase word or a comma | MAUD BM25 recall@10 0.721 → 0.727 |
| Ground truth for ambiguous excerpt pieces used the first match, often a table-of-contents line | "Correct" text contained page numbers ("…Merger 56 SECTION 7.3…") | Choose the match closest to the excerpt's other pieces | TOC-like gold spans: 120 items → 39; MAUD BM25 recall@10 0.752 → 0.792 |

Each fix has a unit test.

## 11. Limitations

- ObliQA questions were generated by an LLM and validated, not written from scratch.
- MAUD covers 2021 US public-company deals; 1.2% of excerpts could not be located.
- The span-overlap threshold (200 characters) is a judgment call; no sensitivity analysis has been run.
- One MAUD query was tuned on test-set failures (with a held-out check). The other 21 MAUD queries were written once and not tuned.
- Results cover retrieval only, not answer quality.

## References

- Gokhan, T., Wang, K., Gurevych, I., & Briscoe, T. (2024). *RIRAG: Regulatory Information Retrieval and Answer Generation.* [arXiv:2409.05677](https://arxiv.org/abs/2409.05677). Dataset: [RegNLP/ObliQADataset](https://github.com/RegNLP/ObliQADataset).
- Wang, S. H., et al. (2023). *MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding.* EMNLP 2023. [arXiv:2301.00876](https://arxiv.org/abs/2301.00876). Dataset: [TheAtticusProject/maud](https://github.com/TheAtticusProject/maud).
