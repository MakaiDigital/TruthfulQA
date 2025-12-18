# src/rag/advanced_rag_system.py
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import Config
from src.rag.query_expansion import QueryExpander
from src.rag.reranker import Reranker
from src.rag.retriever import HybridRetriever
from src.utils.cache import CacheManager


class AdvancedRAGSystem:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        config: Config,
        cache_manager: CacheManager,
    ):
        self.retriever = hybrid_retriever
        self.config = config
        self.cache = cache_manager

        # Initialize components
        self.query_expander = QueryExpander(config)
        self.reranker = Reranker(config.RERANKER_MODEL, config)

        # Initialize LLM
        if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.openai_client = None

        logging.info("Advanced RAG system initialized")

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
        answer_type_filter: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with expansion, hybrid search, and reranking"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(
                "retrieval",
                query=query,
                top_k=top_k,
                source_filter=source_filter,
                answer_type_filter=answer_type_filter,
            )
            if cached is not None:
                logging.info("Retrieved from cache")
                return cached

        # Step 1: Query Expansion
        expanded_queries = self.query_expander.expand(query)
        logging.info(f"Expanded query into {len(expanded_queries)} variants")

        # Step 2: Hybrid retrieval for each expanded query
        all_results = []
        for exp_query in expanded_queries:
            if source_filter or answer_type_filter:
                results = self.retriever.search_with_filter(
                    exp_query,
                    top_k=self.config.RERANK_TOP_K,
                    source_filter=source_filter,
                    answer_type_filter=answer_type_filter,
                )
            else:
                results = self.retriever.hybrid_search(
                    exp_query, top_k=self.config.RERANK_TOP_K
                )
            all_results.extend(results)

        # Step 3: Deduplicate by text
        seen_texts = set()
        unique_results = []
        for result in all_results:
            if result["text"] not in seen_texts:
                seen_texts.add(result["text"])
                unique_results.append(result)

        logging.info(f"Retrieved {len(unique_results)} unique documents")

        # Step 4: Reranking
        k = top_k or self.config.TOP_K_RERANK
        reranked_results = self.reranker.rerank(query, unique_results, top_k=k)

        # Cache results
        if use_cache:
            self.cache.set(
                "retrieval",
                reranked_results,
                query=query,
                top_k=top_k,
                source_filter=source_filter,
                answer_type_filter=answer_type_filter,
            )

        return reranked_results

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API (new API)"""
        if not self.openai_client:
            return self._mock_generate([])

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a truthful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return self._mock_generate([])

    def generate_answer(
        self, query: str, contexts: List[Dict[str, Any]], use_cache: bool = True
    ) -> str:
        """Generate answer using LLM with retrieved context"""

        # Check cache
        if use_cache:
            context_texts = [c["text"] for c in contexts]
            cached = self.cache.get(
                "generation", query=query, contexts=tuple(context_texts)
            )
            if cached is not None:
                logging.info("Retrieved answer from cache")
                return cached

        # Build enhanced context string
        context_str = self._build_context_string(contexts)

        # Create prompt
        prompt = self._create_prompt(query, context_str)

        try:
            if self.config.LLM_PROVIDER == "openai" and self.config.OPENAI_API_KEY:
                answer = self._generate_openai(prompt)
            else:
                answer = self._mock_generate(contexts)

            # Cache answer
            if use_cache:
                context_texts = [c["text"] for c in contexts]
                self.cache.set(
                    "generation", answer, query=query, contexts=tuple(context_texts)
                )

            return answer

        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return self._mock_generate(contexts)

    def _build_context_string(self, contexts: List[Dict[str, Any]]) -> str:
        """Build formatted context with metadata"""
        context_parts = []
        for i, ctx in enumerate(contexts):
            metadata = ctx.get("metadata", {})
            context_part = f"""Context {i+1}:
Text: {ctx['text']}
Related Question: {metadata.get('question', 'N/A')}
Source: {metadata.get('source', 'N/A')}
Relevance Score: {ctx.get('rerank_score', ctx.get('score', 0)):.3f}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context_str: str) -> str:
        """Create enhanced prompt with instructions"""
        return f"""You are a truthful assistant that answers questions based on verified information.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Be concise and accurate
4. Cite the most relevant context when answering
5. Do not make up information

Context:
{context_str}

Question: {query}

Answer:"""

    def _generate_openai_old(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a truthful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def _mock_generate(self, contexts: List[Dict[str, Any]]) -> str:
        """Fallback mock generator"""
        if not contexts:
            return "I don't have enough information to answer this question."

        # Return top context with metadata
        top_context = contexts[0]
        return f"{top_context['text']}"

    def query(
        self,
        question: str,
        source_filter: Optional[List[str]] = None,
        answer_type_filter: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Full RAG pipeline with all enhancements"""
        logging.info(f"Processing query: {question}")

        # Retrieve with all enhancements
        contexts = self.retrieve_context(
            question,
            source_filter=source_filter,
            answer_type_filter=answer_type_filter,
            use_cache=use_cache,
        )

        # Generate answer
        answer = self.generate_answer(question, contexts, use_cache=use_cache)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "num_contexts": len(contexts),
            "filters": {"source": source_filter, "answer_type": answer_type_filter},
        }

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)
        return results

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear_all()
        logging.info("Cache cleared")
