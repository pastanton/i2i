"""
Embedding utilities for statistical consensus mode.

This module provides text embedding functionality for computing semantic
similarity and variance in embedding space. Supports OpenAI embeddings
with fallback to simple TF-IDF based embeddings.
"""

import os
import asyncio
import math
from typing import List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

from .config import get_embedding_model

load_dotenv()


class EmbeddingProvider:
    """
    Provider for text embeddings.

    Supports OpenAI embeddings (recommended) with fallback to
    simple TF-IDF based embeddings for offline use.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or get_embedding_model()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        """Check if embedding provider is configured."""
        return self.api_key is not None

    async def embed(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            numpy array of embedding dimensions
        """
        if self.is_configured():
            return await self._openai_embed(text)
        else:
            return self._fallback_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays
        """
        if self.is_configured():
            return await self._openai_embed_batch(texts)
        else:
            return [self._fallback_embed(t) for t in texts]

    async def _openai_embed(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return np.array(response.data[0].embedding)

    async def _openai_embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get batch embeddings from OpenAI API."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [np.array(d.embedding) for d in response.data]

    def _fallback_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Simple fallback embedding using word hashing.

        This is a basic embedding that uses word hashing to create
        a fixed-dimension vector. Not as good as neural embeddings
        but works offline.
        """
        # Normalize and tokenize
        words = text.lower().split()

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'whom', 'whose', 'my', 'your', 'his', 'her', 'its',
            'our', 'their'
        }
        words = [w for w in words if w not in stop_words and len(w) > 2]

        # Create embedding via word hashing
        embedding = np.zeros(dim)
        for word in words:
            # Hash word to multiple indices for better distribution
            for i in range(3):
                h = hash(word + str(i)) % dim
                embedding[h] += 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two vectors."""
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the centroid (mean) of a list of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Centroid vector
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")
    return np.mean(embeddings, axis=0)


def compute_variance(
    embeddings: List[np.ndarray],
    centroid: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the variance of embeddings around the centroid.

    Uses mean squared distance from centroid.

    Args:
        embeddings: List of embedding vectors
        centroid: Optional pre-computed centroid

    Returns:
        Variance (mean squared distance from centroid)
    """
    if not embeddings:
        return 0.0

    if centroid is None:
        centroid = compute_centroid(embeddings)

    distances_sq = [euclidean_distance(e, centroid) ** 2 for e in embeddings]
    return sum(distances_sq) / len(distances_sq)


def compute_std_dev(
    embeddings: List[np.ndarray],
    centroid: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the standard deviation of embeddings around the centroid.

    Args:
        embeddings: List of embedding vectors
        centroid: Optional pre-computed centroid

    Returns:
        Standard deviation (sqrt of variance)
    """
    return math.sqrt(compute_variance(embeddings, centroid))


def find_outliers(
    embeddings: List[np.ndarray],
    centroid: Optional[np.ndarray] = None,
    threshold: float = 2.0,
) -> List[int]:
    """
    Find indices of outlier embeddings.

    An embedding is an outlier if its distance from the centroid
    exceeds threshold * standard_deviation.

    Args:
        embeddings: List of embedding vectors
        centroid: Optional pre-computed centroid
        threshold: Number of standard deviations for outlier detection

    Returns:
        List of indices of outlier embeddings
    """
    if len(embeddings) < 3:
        return []  # Need at least 3 for meaningful outlier detection

    if centroid is None:
        centroid = compute_centroid(embeddings)

    std_dev = compute_std_dev(embeddings, centroid)
    if std_dev == 0:
        return []

    outliers = []
    for i, emb in enumerate(embeddings):
        distance = euclidean_distance(emb, centroid)
        if distance > threshold * std_dev:
            outliers.append(i)

    return outliers


def find_representative(
    embeddings: List[np.ndarray],
    centroid: Optional[np.ndarray] = None,
) -> int:
    """
    Find the index of the embedding closest to the centroid.

    This represents the most "average" or representative response.

    Args:
        embeddings: List of embedding vectors
        centroid: Optional pre-computed centroid

    Returns:
        Index of the representative embedding
    """
    if not embeddings:
        raise ValueError("Cannot find representative of empty list")

    if centroid is None:
        centroid = compute_centroid(embeddings)

    distances = [euclidean_distance(e, centroid) for e in embeddings]
    return int(np.argmin(distances))


def compute_weighted_centroid(
    centroids: List[np.ndarray],
    variances: List[float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute inverse-variance weighted centroid across models.

    Models with lower variance (more consistent) get higher weight.

    Args:
        centroids: List of per-model centroids
        variances: List of per-model variances
        epsilon: Small value to prevent division by zero

    Returns:
        Weighted centroid
    """
    if not centroids:
        raise ValueError("Cannot compute weighted centroid of empty list")

    # Compute inverse-variance weights
    weights = [1.0 / (v + epsilon) for v in variances]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Weighted sum
    weighted_sum = np.zeros_like(centroids[0])
    for centroid, weight in zip(centroids, normalized_weights):
        weighted_sum += centroid * weight

    return weighted_sum


def consistency_score(std_dev: float) -> float:
    """
    Convert standard deviation to a consistency score in [0, 1].

    Higher score = more consistent (lower variance).

    Args:
        std_dev: Standard deviation of embeddings

    Returns:
        Consistency score between 0 and 1
    """
    return 1.0 / (1.0 + std_dev)
