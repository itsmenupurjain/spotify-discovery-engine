"""
RAG Query Engine — natural language query interface over structured + vector data.
Implements spec §6.3: embed query → vector search → structured filters → AI synthesis.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.ai.embedder import Embedder
from app.config import settings
from app.models.classified_review import ClassifiedReview
from app.models.raw_review import RawReview
from app.models.query_log import QueryLog

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """SYSTEM: You are a product research analyst at Spotify.
Answer the PM's question based ONLY on the provided user feedback data.
Be specific, cite evidence, and structure your answer clearly.
If the data doesn't contain enough information to answer confidently, say so.

USER QUESTION: {query}

RELEVANT FEEDBACK DATA:
{data}

Respond with a JSON object:
{{
  "answer": "string (2-3 sentences, direct answer)",
  "evidence": [
    {{"text": "quote or finding", "source": "source name", "date": "date if available"}}
  ],
  "confidence": "High / Medium / Low",
  "confidence_reason": "string (why this confidence level)",
  "follow_up_questions": ["string", "string", "string"]
}}"""


class RAGQueryEngine:
    """RAG-powered natural language query engine for PM dashboard."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = Embedder()

    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query:
        1. Embed the query
        2. Vector similarity search (top-k=20)
        3. NLP intent extraction for structured filters
        4. Hybrid retrieval (vector + SQL)
        5. AI synthesis
        6. Log query

        Returns structured response with answer, evidence, confidence, and follow-ups.
        """
        start_time = time.time()

        # 1. Embed query
        query_embedding = await self.embedder.embed_single(query_text)

        # 2. Vector similarity search
        vector_results = []
        if query_embedding:
            vector_results = await self._vector_search(query_embedding, top_k=20)

        # 3. Extract intent for structured filters
        filters = self._extract_intent(query_text)

        # 4. Structured SQL search (if filters detected)
        sql_results = await self._structured_search(filters)

        # 5. Merge and deduplicate results
        merged = self._merge_results(vector_results, sql_results)

        # 6. AI synthesis
        synthesis = await self._synthesize(query_text, merged)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # 7. Log query
        await self._log_query(query_text, query_embedding, len(merged), response_time_ms)

        return {
            "answer": synthesis.get("answer", "Unable to generate answer."),
            "evidence": synthesis.get("evidence", []),
            "confidence": synthesis.get("confidence", "Low"),
            "confidence_reason": synthesis.get("confidence_reason", ""),
            "follow_up_questions": synthesis.get("follow_up_questions", []),
            "result_count": len(merged),
            "query_time_ms": response_time_ms,
            "raw_results": merged[:10],  # Return top 10 for UI display
        }

    async def _vector_search(self, embedding: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
        """Cosine similarity search over classified_reviews embeddings."""
        try:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            query = text(f"""
                SELECT
                    cr.id, cr.raw_review_id, cr.primary_complaint_category,
                    cr.user_segment_signal, cr.sentiment, cr.sentiment_score,
                    cr.key_frustration_phrase, cr.unmet_need, cr.jtbd_statement,
                    cr.confidence_score, cr.source_weight, cr.recency_score,
                    rr.source, rr.body, rr.rating, rr.published_at, rr.raw_url,
                    1 - (cr.embedding <=> '{embedding_str}'::vector) as similarity
                FROM classified_reviews cr
                JOIN raw_reviews rr ON cr.raw_review_id = rr.id
                WHERE cr.embedding IS NOT NULL
                AND cr.classification_failed = false
                ORDER BY cr.embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)

            result = await self.db.execute(query, {"top_k": top_k})
            rows = result.fetchall()

            return [
                {
                    "id": str(row[0]),
                    "source": row[12],
                    "body": row[13][:500] if row[13] else "",
                    "rating": row[14],
                    "published_at": row[15].isoformat() if row[15] else None,
                    "primary_complaint_category": row[2],
                    "user_segment_signal": row[3],
                    "sentiment": row[4],
                    "key_frustration_phrase": row[6],
                    "unmet_need": row[7],
                    "similarity_score": float(row[17]) if row[17] else 0,
                    "retrieval_method": "vector",
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _extract_intent(self, query: str) -> Dict[str, Any]:
        """Simple NLP intent extraction from query text."""
        query_lower = query.lower()
        filters = {}

        # Source detection
        source_keywords = {
            "reddit": "reddit",
            "app store": "app_store",
            "play store": "play_store",
            "google play": "play_store",
            "community": "spotify_community",
            "forum": "spotify_community",
            "twitter": "twitter_x",
            "tweet": "twitter_x",
        }
        for keyword, source in source_keywords.items():
            if keyword in query_lower:
                filters["source"] = source
                break

        # Segment detection
        segment_keywords = {
            "explorer": "active_explorer_stuck",
            "stuck": "active_explorer_stuck",
            "background": "background_listener",
            "mood": "mood_regulator",
            "identity": "identity_listener",
            "social": "socially_led_discoverer",
            "new user": "new_user",
        }
        for keyword, segment in segment_keywords.items():
            if keyword in query_lower:
                filters["segment"] = segment
                break

        # Category detection
        category_keywords = {
            "staleness": "algorithm_staleness",
            "stale": "algorithm_staleness",
            "algorithm": "algorithm_staleness",
            "overload": "choice_overload",
            "overwhelm": "choice_overload",
            "trust": "trust_erosion",
            "mood": "mood_mismatch",
            "interface": "interface_friction",
            "context": "lack_of_context",
            "bubble": "genre_bubble",
            "genre": "genre_bubble",
            "social": "social_disconnect",
        }
        for keyword, category in category_keywords.items():
            if keyword in query_lower:
                filters["category"] = category
                break

        return filters

    async def _structured_search(self, filters: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """Structured SQL search based on extracted filters."""
        if not filters:
            return []

        query = (
            select(ClassifiedReview, RawReview)
            .join(RawReview, ClassifiedReview.raw_review_id == RawReview.id)
            .where(ClassifiedReview.classification_failed == False)
        )

        if filters.get("source"):
            query = query.where(RawReview.source == filters["source"])
        if filters.get("segment"):
            query = query.where(ClassifiedReview.user_segment_signal == filters["segment"])
        if filters.get("category"):
            query = query.where(ClassifiedReview.primary_complaint_category == filters["category"])

        query = query.order_by(RawReview.published_at.desc()).limit(limit)

        result = await self.db.execute(query)
        rows = result.all()

        return [
            {
                "id": str(classified.id),
                "source": raw.source,
                "body": raw.body[:500] if raw.body else "",
                "rating": raw.rating,
                "published_at": raw.published_at.isoformat() if raw.published_at else None,
                "primary_complaint_category": classified.primary_complaint_category,
                "user_segment_signal": classified.user_segment_signal,
                "sentiment": classified.sentiment,
                "key_frustration_phrase": classified.key_frustration_phrase,
                "unmet_need": classified.unmet_need,
                "similarity_score": 0.0,
                "retrieval_method": "structured",
            }
            for classified, raw in rows
        ]

    def _merge_results(
        self, vector_results: List[Dict], sql_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from vector + structured search."""
        seen_ids = set()
        merged = []

        # Vector results first (higher relevance)
        for r in vector_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        # Then SQL results
        for r in sql_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)

        return merged

    async def _synthesize(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to synthesize results into a structured answer."""
        if not results:
            return {
                "answer": "No relevant data found for your query. Try rephrasing or broadening your search.",
                "evidence": [],
                "confidence": "Low",
                "confidence_reason": "No matching reviews found",
                "follow_up_questions": [
                    "What are the top themes across all sources?",
                    "Show me the most common complaints",
                    "What do users want from music discovery?",
                ],
            }

        # Format data for AI
        data_str = json.dumps(results[:20], indent=2, default=str)
        prompt = SYNTHESIS_PROMPT.format(query=query, data=data_str)

        try:
            # Use Groq via OpenAI SDK
            if settings.groq_api_key:
                import openai
                client = openai.AsyncOpenAI(
                    api_key=settings.groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                response = await client.chat.completions.create(
                    model=settings.classification_model,
                    temperature=0.3,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq synthesis failed: {e}")

        # Manual fallback
        return {
            "answer": f"Found {len(results)} relevant reviews. AI synthesis unavailable — review the raw results below.",
            "evidence": [
                {"text": r.get("key_frustration_phrase", r.get("body", "")[:100]), "source": r.get("source", "")}
                for r in results[:5]
            ],
            "confidence": "Low",
            "confidence_reason": "AI synthesis unavailable",
            "follow_up_questions": [],
        }

    async def _log_query(self, query_text: str, embedding: Optional[List[float]], result_count: int, response_time_ms: int):
        """Log query for analytics."""
        try:
            log_entry = QueryLog(
                query_text=query_text,
                result_count=result_count,
                response_time_ms=response_time_ms,
            )
            # Note: We skip storing the embedding in the log to save space
            self.db.add(log_entry)
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
