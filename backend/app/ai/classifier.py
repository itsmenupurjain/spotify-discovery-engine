"""
AI Classifier — classifies reviews using Claude API (primary) with GPT-4o fallback.
Implements the structured classification prompt from spec §4.1.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """SYSTEM:
You are a product research analyst at Spotify working on the Growth Team.
Your job is to classify user feedback to understand why users struggle to discover new music.
You must respond ONLY with a valid JSON object. No preamble, no explanation, no markdown.

USER:
Analyze the following user feedback and return a JSON object with these exact fields:

INPUT TEXT:
"{text}"

SOURCE: {source}
RATING (if available): {rating}

REQUIRED OUTPUT JSON:
{{
  "is_discovery_relevant": true/false,
  "primary_complaint_category": "one of: [algorithm_staleness, choice_overload, trust_erosion, mood_mismatch, interface_friction, lack_of_context, genre_bubble, social_disconnect, no_complaint, other]",
  "secondary_complaint_category": "same options as above or null",
  "user_segment_signal": "one of: [active_explorer_stuck, background_listener, mood_regulator, identity_listener, socially_led_discoverer, new_user, unclear]",
  "sentiment": "one of: [very_negative, negative, neutral, positive, very_positive]",
  "sentiment_score": float between -1.0 and 1.0,
  "discovery_intent": "one of: [high, medium, low, none]",
  "repetition_behavior_mentioned": true/false,
  "key_frustration_phrase": "string — the single most important phrase from the text that captures the core frustration, or null",
  "unmet_need": "string — a concise (max 15 words) statement of what the user actually wants but isn't getting, or null",
  "jtbd_statement": "string — complete this: 'When [situation], I want [need], so that [outcome]' based on this text, or null",
  "confidence_score": float between 0.0 and 1.0,
  "language": "ISO 639-1 language code"
}}

CLASSIFICATION DEFINITIONS:
- algorithm_staleness: User feels recommendations don't change or improve over time
- choice_overload: User feels overwhelmed by too many options or can't decide what to try
- trust_erosion: User tried recommendations, didn't like them, stopped trusting the system
- mood_mismatch: Recommendations don't match the user's current mood or context
- interface_friction: Discovery surfaces are hard to find, navigate, or use
- lack_of_context: Algorithm doesn't understand WHY the user listens to certain songs (context collapse)
- genre_bubble: Algorithm is too narrow, keeps user in one genre/era/artist cluster
- social_disconnect: User discovers music through humans/social media, not the algorithm
- no_complaint: Text is positive or neutral with no frustration signal
- other: Frustration present but doesn't fit above categories

USER SEGMENT DEFINITIONS:
- active_explorer_stuck: User wants to discover new music but keeps repeating familiar content
- background_listener: Music is functional/background; discovery is not important to them
- mood_regulator: Uses music to achieve a specific emotional state; new music is risky
- identity_listener: Music reflects identity/nostalgia; discovery irrelevant to their job
- socially_led_discoverer: Discovers through friends/social, not algorithm
- new_user: Recently joined, still building library
- unclear: Cannot determine segment from text"""


class AIClassifier:
    """Classifies reviews using Claude API with GPT-4o fallback."""

    def __init__(self):
        self.primary_model = settings.classification_model
        self.temperature = settings.classification_temperature
        self.max_tokens = settings.classification_max_tokens
        self.batch_size = settings.classification_batch_size

    async def classify_batch(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of review entries.

        Args:
            entries: List of dicts with 'body', 'source', 'rating' fields.

        Returns:
            List of classification result dicts.
        """
        results = []

        for i in range(0, len(entries), self.batch_size):
            batch = entries[i:i + self.batch_size]
            logger.info(f"Classifying batch {i // self.batch_size + 1} ({len(batch)} entries)...")

            for entry in batch:
                try:
                    result = await self._classify_single(entry)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Classification failed for entry: {e}")
                    results.append({
                        "classification_failed": True,
                        "error": str(e),
                        "raw_review_id": entry.get("id"),
                    })

        return results

    async def _classify_single(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single review entry using Claude API with fallback."""
        text = entry.get("body", "")
        source = entry.get("source", "unknown")
        rating = entry.get("rating", "N/A")

        prompt = CLASSIFICATION_PROMPT.format(
            text=text[:2000],  # Truncate to 2000 chars
            source=source,
            rating=rating,
        )

        # Try Groq API
        result = await self._call_groq(prompt)
        if result:
            result["classification_model"] = self.primary_model
            result["raw_review_id"] = entry.get("id")
            return result

        # Failed
        return {
            "classification_failed": True,
            "error": "Groq classification failed",
            "raw_review_id": entry.get("id"),
        }

    async def _call_groq(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Groq API via OpenAI SDK for classification."""
        if not settings.groq_api_key:
            return None

        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            response = await client.chat.completions.create(
                model=self.primary_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            return self._parse_json_response(response_text)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response with retry logic."""
        try:
            # Try direct parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            try:
                json_match = response_text.split("```json")[-1].split("```")[0]
                return json.loads(json_match)
            except (json.JSONDecodeError, IndexError):
                # Try finding JSON object in text
                try:
                    start = response_text.index("{")
                    end = response_text.rindex("}") + 1
                    return json.loads(response_text[start:end])
                except (ValueError, json.JSONDecodeError):
                    logger.error(f"Failed to parse JSON from response: {response_text[:200]}")
                    return None
