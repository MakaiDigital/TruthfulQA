import logging
from typing import List

import nltk
import openai
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


class QueryExpander:
    def __init__(self, config):
        self.config = config
        self.llm_available = bool(config.OPENAI_API_KEY)
        if self.llm_available:
            openai.api_key = config.OPENAI_API_KEY

    def expand_query_llm(self, query: str, num_expansions: int = 2) -> List[str]:
        """Expand query using LLM to generate alternative phrasings"""
        if not self.llm_available:
            return [query]

        try:
            prompt = f"""Generate {num_expansions} alternative ways to ask the following question. 
Keep the same meaning but use different words and phrasings.
Only output the alternative questions, one per line.

Original question: {query}

Alternative questions:"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that rephrases questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=150,
            )

            expansions = response.choices[0].message.content.strip().split("\n")
            expansions = [e.strip() for e in expansions if e.strip()]

            # Return original + expansions
            return [query] + expansions[:num_expansions]

        except Exception as e:
            logging.error(f"LLM query expansion failed: {e}")
            return self.expand_query_wordnet(query)

    def expand_query_wordnet(self, query: str) -> List[str]:
        """Expand query using WordNet synonyms (fallback method)"""
        words = query.lower().split()
        expanded_queries = [query]

        # Get synonyms for key words
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                # Get first synonym
                synonym = synsets[0].lemmas()[0].name().replace("_", " ")
                if synonym != word:
                    expanded_query = query.replace(word, synonym)
                    expanded_queries.append(expanded_query)

        return list(set(expanded_queries))[:3]  # Limit to 3 total

    def expand(self, query: str) -> List[str]:
        """Main expansion method"""
        if not self.config.ENABLE_QUERY_EXPANSION:
            return [query]

        if self.llm_available:
            return self.expand_query_llm(query, self.config.NUM_EXPANSIONS)
        else:
            return self.expand_query_wordnet(query)
