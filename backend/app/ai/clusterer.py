"""
Theme Clusterer — k-means clustering on frustration phrase embeddings + AI theme labeling.
Implements spec §4.2.
"""

import logging
import json
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.ai.embedder import Embedder
from app.config import settings

logger = logging.getLogger(__name__)

THEME_LABELING_PROMPT = """SYSTEM: You are a product researcher. Given these user feedback excerpts that have been grouped together, identify the single unifying theme they share. Respond with a valid JSON object only:
{{"theme_name": "string (3-5 words)", "theme_description": "string (1 sentence)", "representative_quote": "string (best single quote from the list)"}}

USER: Here are {n} user feedback excerpts grouped together:
{excerpts}"""


class ThemeClusterer:
    """Clusters reviews by frustration phrase embeddings and labels themes via AI."""

    def __init__(self):
        self.embedder = Embedder()
        self.initial_k = settings.initial_cluster_count
        self.representatives_per_cluster = settings.cluster_representatives

    async def cluster_and_label(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Full clustering pipeline:
        1. Embed frustration phrases
        2. K-means clustering (optimized k via silhouette score)
        3. AI-based theme labeling per cluster
        4. Return theme objects + review mappings

        Args:
            entries: List of classified reviews with 'key_frustration_phrase' and 'id'.

        Returns:
            List of theme dicts with 'cluster_id', 'theme_name', 'theme_description',
            'representative_quote', and 'review_ids'.
        """
        # Filter entries with frustration phrases
        valid_entries = [
            e for e in entries
            if e.get("key_frustration_phrase") and len(e["key_frustration_phrase"].strip()) > 10
        ]

        if len(valid_entries) < self.initial_k:
            logger.warning(f"Not enough entries for clustering ({len(valid_entries)} < {self.initial_k})")
            return []

        # Step 1: Embed frustration phrases
        phrases = [e["key_frustration_phrase"] for e in valid_entries]
        embeddings = await self.embedder.embed_texts(phrases)

        # Filter out None embeddings
        valid_pairs = [
            (entry, emb) for entry, emb in zip(valid_entries, embeddings)
            if emb is not None
        ]

        if len(valid_pairs) < self.initial_k:
            logger.warning("Not enough valid embeddings for clustering")
            return []

        entries_filtered = [p[0] for p in valid_pairs]
        embedding_matrix = np.array([p[1] for p in valid_pairs])

        # Step 2: Find optimal k via silhouette score
        optimal_k = self._find_optimal_k(embedding_matrix)

        # Step 3: K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        # Step 4: Group entries by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                **entries_filtered[idx],
                "embedding_idx": idx,
            })

        # Step 5: Label each cluster via AI
        themes = []
        for cluster_id, cluster_entries in clusters.items():
            # Get top N representative entries (closest to centroid)
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = [
                np.linalg.norm(embedding_matrix[e["embedding_idx"]] - centroid)
                for e in cluster_entries
            ]
            sorted_entries = [e for _, e in sorted(zip(distances, cluster_entries))]
            representatives = sorted_entries[:self.representatives_per_cluster]

            # AI labeling
            theme = await self._label_cluster(cluster_id, representatives)
            if theme:
                theme["review_ids"] = [e.get("id") for e in cluster_entries]
                theme["review_count"] = len(cluster_entries)

                # Count distinct sources
                sources = set(e.get("source") for e in cluster_entries if e.get("source"))
                theme["cross_source_count"] = len(sources)
                theme["confidence_level"] = (
                    "high" if len(sources) >= 3
                    else "medium" if len(sources) >= 2
                    else "low"
                )

                themes.append(theme)

        logger.info(f"Identified {len(themes)} themes from {len(entries_filtered)} entries (k={optimal_k})")
        return themes

    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """Find optimal k using silhouette score, starting from initial_k."""
        best_k = self.initial_k
        best_score = -1

        k_range = range(max(3, self.initial_k - 3), self.initial_k + 4)

        for k in k_range:
            if k >= len(embeddings):
                continue
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))

                if score > best_score:
                    best_score = score
                    best_k = k

                logger.debug(f"k={k}, silhouette={score:.3f}")
            except Exception:
                continue

        logger.info(f"Optimal k={best_k} (silhouette={best_score:.3f})")
        return best_k

    async def _label_cluster(self, cluster_id: int, representatives: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use AI to generate a theme label for a cluster."""
        excerpts = "\n".join([
            f"- \"{r.get('key_frustration_phrase', '')}\""
            for r in representatives
        ])

        prompt = THEME_LABELING_PROMPT.format(
            n=len(representatives),
            excerpts=excerpts,
        )

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
                    temperature=0.2,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content
                result = json.loads(response_text)
                result["cluster_id"] = cluster_id
                return result

        except Exception as e:
            logger.error(f"Theme labeling failed for cluster {cluster_id}: {e}")

        # Manual fallback
        return {
            "cluster_id": cluster_id,
            "theme_name": f"Theme {cluster_id + 1}",
            "theme_description": "Theme auto-generated — AI labeling unavailable",
            "representative_quote": representatives[0].get("key_frustration_phrase", ""),
        }
