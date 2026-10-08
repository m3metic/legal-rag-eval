#!/usr/bin/env bash
# Download both datasets into data/raw/. They are not committed to this repo:
# check each dataset's terms and cite the authors (see README) if you use them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if [ ! -d data/raw/ObliQADataset ]; then
  git clone --depth 1 https://github.com/RegNLP/ObliQADataset.git data/raw/ObliQADataset
fi

if [ ! -d data/raw/maud/data/contracts ]; then
  git clone --depth 1 https://github.com/TheAtticusProject/maud.git data/raw/maud
  (cd data/raw/maud && unzip -q -o data.zip)
fi

echo "ObliQA documents: $(ls data/raw/ObliQADataset/StructuredRegulatoryDocuments/*.json | wc -l)"
echo "MAUD contracts:   $(ls data/raw/maud/data/contracts/*.txt | wc -l)"
