import json
import logging
import re
from typing import Any, Dict, List

import pandas as pd


class DataProcessor:
    def __init__(self):
        self._logged_columns = False

    def extract_truthful_answers(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Extract only correct answers, filtering out incorrect ones"""
        documents = []

        # Log columns once for debugging
        if not self._logged_columns:
            logging.info(f"Dataset columns: {list(row.index)}")
            self._logged_columns = True

        # Get question
        question = row.get("Question")
        if pd.isna(question) or not str(question).strip():
            return documents

        question = str(question).strip()

        # Extract Best Answer
        best_answer = row.get("Best Answer")
        if pd.notna(best_answer):
            best_answer_text = str(best_answer).strip()
            if best_answer_text:
                documents.append(
                    {
                        "text": best_answer_text,
                        "question": question,
                        "answer_type": "best_answer",
                        "source": str(row.get("Source", "TruthfulQA")),
                    }
                )

        # Extract Correct Answers (semicolon-separated in TruthfulQA)
        correct_answers_field = row.get("Correct Answers")
        if pd.notna(correct_answers_field):
            correct_answers = self._parse_answers(correct_answers_field)
            for idx, answer in enumerate(correct_answers):
                answer_text = answer.strip()
                if answer_text:
                    documents.append(
                        {
                            "text": answer_text,
                            "question": question,
                            "answer_type": f"correct_answer_{idx}",
                            "source": str(row.get("Source", "TruthfulQA")),
                        }
                    )

        #  We explicitly DO NOT extract from 'Incorrect Answers' column
        # This ensures only truthful information goes into the knowledge base

        return documents

    def _parse_answers(self, answers_field) -> List[str]:
        """Parse answer field - handles semicolon-separated strings"""
        if pd.isna(answers_field):
            return []

        # Convert to string
        answers_str = str(answers_field).strip()

        if not answers_str:
            return []

        # TruthfulQA uses semicolons to separate multiple answers
        if ";" in answers_str:
            answers = [a.strip() for a in answers_str.split(";")]
            return [a for a in answers if a]

        # Fallback: treat as single answer
        return [answers_str]

    def filter_and_clean(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process entire dataset, excluding incorrect answers"""
        all_documents = []

        logging.info(f"Processing {len(df)} questions...")

        for idx, row in df.iterrows():
            docs = self.extract_truthful_answers(row)
            all_documents.extend(docs)

            # Log progress every 200 rows
            if (idx + 1) % 200 == 0:
                logging.info(
                    f"  Processed {idx + 1}/{len(df)} questions -> {len(all_documents)} documents"
                )

        logging.info(f"Total documents extracted: {len(all_documents)}")

        if len(all_documents) == 0:
            logging.error(" No documents extracted!")
            logging.error("Sample of first row:")
            if len(df) > 0:
                for col in df.columns:
                    value = df[col].iloc[0]
                    logging.error(f"  {col}: {str(value)[:100]}")
            return []

        # Remove duplicates based on (question, text) pair
        seen = set()
        unique_docs = []
        for doc in all_documents:
            key = (doc["question"], doc["text"])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        logging.info(f"Unique documents after deduplication: {len(unique_docs)}")

        return unique_docs
