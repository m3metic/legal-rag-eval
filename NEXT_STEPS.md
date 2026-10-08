# Next steps (your roadmap — delete this file before you make the repo public)

The engine, both datasets and the BM25 baselines are done and tested. What's
left is the part that turns this into *your* project: running the
experiments, digging into failures, and writing up what you learned.
Interviewers will ask about these decisions, so make each one yourself and
keep notes as you go.

## 1. Get it running locally (~30 min)
- [ ] `pip install -e ".[dense,dev]"`, `./scripts/download_data.sh`, `pytest`
- [ ] Re-run `configs/obliqa/bm25_passage.yaml` and check you get recall@10 ≈ 0.754
- [ ] Read `metrics.py` and `datasets/maud.py` until you could explain them on a whiteboard.
      Likely interview questions: *why recall at a word budget? how do you score a
      chunk against a span? what happens to the 34 MAUD items you couldn't locate?*

## 2. Dense, hybrid and reranking runs (~an evening)
- [ ] Run `configs/obliqa/dense_*.yaml` and `hybrid_*.yaml`. Embedding 13k passages with
      bge-small takes a few minutes on a laptop CPU; results are cached.
- [ ] Same for `configs/maud/`. MAUD is ~50k chunks, so try `--limit 300` first.
- [ ] Try `hybrid_rerank_passage_header.yaml` (slower: a cross-encoder scores 50 candidates per query).
- [ ] Add one stronger embedding model of your choice and compare. Note the
      prefix it needs (check its model card).

## 3. Error analysis (the most impressive section of the write-up)
- [ ] Open `results/maud_bm25_section300/per_query.jsonl`. Pick 10 failures from
      "Negative interim operating covenant" and read what was retrieved instead. Why?
- [ ] Same for ObliQA multi-passage questions. Are the missing passages in the same
      section as the found ones? (If so, "expand to neighbouring passages" is an experiment.)
- [ ] Write 3–5 bullet findings with an example each.

## 4. Your own improvements (pick one or two)
- [ ] **Rewrite the MAUD queries** in `configs/maud_queries.yaml`. The label-vs-question
      result shows wording matters a lot. Try lawyer phrasing, keyword lists, or multiple
      queries per deal point fused with RRF.
- [ ] **Query expansion / HyDE**: have a small LLM (e.g. via Ollama) write a hypothetical
      clause, then search with it.
- [ ] **Neighbour expansion** for ObliQA: after retrieving a passage, also return its siblings.
- [ ] **Fine-tune the embedding model** on the ObliQA train split (22k questions) and test
      on the test split. Never tune anything on the test split.
- [ ] Swap `rank_bm25` for `bm25s` to make BM25 runs ~10× faster.

## 5. Add answer generation (phase 2)
- [ ] Generate answers from the top chunks with a local model, and cite chunk IDs.
- [ ] For MAUD, answer the multiple-choice deal-point questions in the CSVs and measure
      accuracy end to end. That shows how retrieval errors turn into answer errors.

## 6. Polish for GitHub
- [ ] Regenerate the README tables with `python -m legalrag compare ...`
- [ ] Add one chart (e.g. recall@1000w by chunking strategy, or recall by deal point)
- [ ] Rewrite the README findings in your own words
- [ ] Optional: a small Streamlit demo where you pick a contract and a deal point
      and see what each retriever returns
