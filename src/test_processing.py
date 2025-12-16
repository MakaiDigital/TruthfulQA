# test_processing.py
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.ingestion import DataIngestion
from src.pipeline.processing import DataProcessor

print("Testing data processing...\n")

# Load data
ingestion = DataIngestion(Path("data"))
df = ingestion.load_dataset()
print(f" Loaded {len(df)} questions\n")

# Process
processor = DataProcessor()

# Test on first 5 rows
print("Testing on first 5 rows:")
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    docs = processor.extract_truthful_answers(row)
    print(f"\nRow {idx + 1}:")
    print(f"  Question: {row['Question'][:60]}...")
    print(f"  Extracted {len(docs)} documents:")
    for doc in docs:
        print(f"    - [{doc['answer_type']}] {doc['text'][:80]}...")

# Process all
print("\n" + "=" * 60)
print("Processing full dataset...")
all_docs = processor.filter_and_clean(df)
print(f"✓ Extracted {len(all_docs)} total documents")

if all_docs:
    print("\nSample documents:")
    for i, doc in enumerate(all_docs[:3]):
        print(f"\n{i+1}. Question: {doc['question'][:60]}...")
        print(f"   Answer: {doc['text'][:80]}...")
        print(f"   Type: {doc['answer_type']}")
else:
    print("\n No documents extracted!")
