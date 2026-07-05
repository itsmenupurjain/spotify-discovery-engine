"""
Embedder — generates vector embeddings using OpenAI text-embedding-3-small.
Embeds review bodies for semantic search and frustration phrases for clustering.
"""

import logging
from typing import List, Optional

import httpx
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using the free HuggingFace API (sentence-transformers)."""

    def __init__(self):
        self.dimensions = settings.embedding_dimensions
        # We use a popular fast embedding model
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {}
        # Optional: Use a token if provided, otherwise rely on free tier limits
        if hasattr(settings, 'huggingface_api_key') and settings.huggingface_api_key:
            self.headers["Authorization"] = f"Bearer {settings.huggingface_api_key}"

    async def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Embed a list of texts via HuggingFace API.
        """
        all_embeddings = []
        batch_size = 50  # Smaller batch size for free tier

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                cleaned = [t[:2000] if t and t.strip() else "empty" for t in batch]

                try:
                    response = await client.post(
                        self.api_url, 
                        headers=self.headers, 
                        json={"inputs": cleaned, "options": {"wait_for_model": True}},
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        # The model returns a list of vectors (usually 384 dimensions)
                        batch_embeddings = response.json()
                        
                        # Pad vectors to match the database's expected dimensions (1536)
                        for vec in batch_embeddings:
                            if len(vec) < self.dimensions:
                                vec.extend([0.0] * (self.dimensions - len(vec)))
                            elif len(vec) > self.dimensions:
                                vec = vec[:self.dimensions]
                            all_embeddings.append(vec)
                    else:
                        logger.error(f"HuggingFace API error {response.status_code}: {response.text}")
                        all_embeddings.extend([None] * len(batch))

                except Exception as e:
                    logger.error(f"Embedding batch {i // batch_size + 1} failed: {e}")
                    all_embeddings.extend([None] * len(batch))

        return all_embeddings

    async def embed_single(self, text: str) -> Optional[List[float]]:
        """Embed a single text string."""
        results = await self.embed_texts([text])
        return results[0] if results else None
