"""Tests for i2i embeddings module."""

import pytest
import numpy as np

from i2i.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    compute_centroid,
    compute_variance,
    compute_std_dev,
    find_outliers,
    find_representative,
    compute_weighted_centroid,
    consistency_score,
)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors_have_similarity_one(self):
        """Identical vectors should have similarity of 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        """Orthogonal vectors should have similarity of 0.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors_have_similarity_negative_one(self):
        """Opposite vectors should have similarity of -1.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0 similarity."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_similar_vectors_have_high_similarity(self):
        """Similar vectors should have high similarity."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.1, 2.1, 3.1])
        assert cosine_similarity(v1, v2) > 0.99


class TestCosineDistance:
    """Tests for cosine distance function."""

    def test_identical_vectors_have_distance_zero(self):
        """Identical vectors should have distance of 0.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0)

    def test_opposite_vectors_have_distance_two(self):
        """Opposite vectors should have distance of 2.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(2.0)


class TestEuclideanDistance:
    """Tests for Euclidean distance function."""

    def test_identical_vectors_have_distance_zero(self):
        """Identical vectors should have distance of 0.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_unit_vectors_have_correct_distance(self):
        """Unit distance should be correct."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert euclidean_distance(v1, v2) == pytest.approx(1.0)

    def test_pythagorean_distance(self):
        """3-4-5 triangle should have distance 5."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)


class TestComputeCentroid:
    """Tests for centroid computation."""

    def test_single_embedding_returns_itself(self):
        """Single embedding should return itself as centroid."""
        v = np.array([1.0, 2.0, 3.0])
        centroid = compute_centroid([v])
        np.testing.assert_array_equal(centroid, v)

    def test_centroid_of_symmetric_points(self):
        """Centroid of symmetric points should be at origin."""
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
        centroid = compute_centroid(embeddings)
        np.testing.assert_array_almost_equal(centroid, np.array([0.0, 0.0]))

    def test_centroid_is_mean(self):
        """Centroid should be the mean of embeddings."""
        embeddings = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ]
        centroid = compute_centroid(embeddings)
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_equal(centroid, expected)

    def test_empty_list_raises_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            compute_centroid([])


class TestComputeVariance:
    """Tests for variance computation."""

    def test_single_embedding_has_zero_variance(self):
        """Single embedding should have zero variance."""
        v = np.array([1.0, 2.0, 3.0])
        variance = compute_variance([v])
        assert variance == pytest.approx(0.0)

    def test_identical_embeddings_have_zero_variance(self):
        """Identical embeddings should have zero variance."""
        v = np.array([1.0, 2.0, 3.0])
        embeddings = [v, v, v]
        variance = compute_variance(embeddings)
        assert variance == pytest.approx(0.0)

    def test_spread_embeddings_have_positive_variance(self):
        """Spread embeddings should have positive variance."""
        embeddings = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        variance = compute_variance(embeddings)
        assert variance > 0

    def test_empty_list_has_zero_variance(self):
        """Empty list should return zero variance."""
        assert compute_variance([]) == 0.0


class TestComputeStdDev:
    """Tests for standard deviation computation."""

    def test_std_dev_is_sqrt_of_variance(self):
        """Standard deviation should be sqrt of variance."""
        embeddings = [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        ]
        variance = compute_variance(embeddings)
        std_dev = compute_std_dev(embeddings)
        assert std_dev == pytest.approx(np.sqrt(variance))


class TestFindOutliers:
    """Tests for outlier detection."""

    def test_no_outliers_when_all_close(self):
        """No outliers when all embeddings are close."""
        embeddings = [
            np.array([1.0, 1.0]),
            np.array([1.1, 1.0]),
            np.array([1.0, 1.1]),
            np.array([0.9, 1.0]),
        ]
        outliers = find_outliers(embeddings, threshold=2.0)
        assert len(outliers) == 0

    def test_detects_obvious_outlier(self):
        """Should detect an obvious outlier with many normal points."""
        # Need many normal points so the outlier doesn't dominate mean/std
        embeddings = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.0]),
            np.array([0.0, 0.1]),
            np.array([0.1, 0.1]),
            np.array([0.05, 0.05]),
            np.array([0.02, 0.08]),
            np.array([0.08, 0.02]),
            np.array([0.15, 0.15]),
            np.array([0.0, 0.15]),
            np.array([10.0, 10.0]),  # Obvious outlier - index 9
        ]
        outliers = find_outliers(embeddings, threshold=2.0)
        assert 9 in outliers  # Index of the outlier

    def test_returns_empty_for_small_lists(self):
        """Should return empty for lists with fewer than 3 items."""
        embeddings = [
            np.array([0.0, 0.0]),
            np.array([100.0, 100.0]),
        ]
        outliers = find_outliers(embeddings, threshold=2.0)
        assert len(outliers) == 0


class TestFindRepresentative:
    """Tests for finding representative embedding."""

    def test_finds_closest_to_centroid(self):
        """Should find the embedding closest to centroid."""
        embeddings = [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([1.0, 0.0]),  # Closest to centroid (1.0, 0.0)
        ]
        idx = find_representative(embeddings)
        assert idx == 2

    def test_single_embedding_returns_zero(self):
        """Single embedding should return index 0."""
        embeddings = [np.array([5.0, 5.0])]
        idx = find_representative(embeddings)
        assert idx == 0

    def test_empty_list_raises_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            find_representative([])


class TestComputeWeightedCentroid:
    """Tests for inverse-variance weighted centroid."""

    def test_equal_variances_returns_mean(self):
        """Equal variances should return regular mean."""
        centroids = [
            np.array([1.0, 0.0]),
            np.array([3.0, 0.0]),
        ]
        variances = [1.0, 1.0]
        weighted = compute_weighted_centroid(centroids, variances)
        expected = np.array([2.0, 0.0])
        np.testing.assert_array_almost_equal(weighted, expected)

    def test_low_variance_has_higher_weight(self):
        """Low variance centroid should dominate."""
        centroids = [
            np.array([0.0, 0.0]),  # Low variance
            np.array([10.0, 0.0]),  # High variance
        ]
        variances = [0.01, 10.0]  # Very different
        weighted = compute_weighted_centroid(centroids, variances)
        # Should be much closer to first centroid
        assert weighted[0] < 1.0

    def test_empty_list_raises_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            compute_weighted_centroid([], [])


class TestConsistencyScore:
    """Tests for consistency score computation."""

    def test_zero_std_dev_gives_score_one(self):
        """Zero std dev should give score of 1.0."""
        assert consistency_score(0.0) == pytest.approx(1.0)

    def test_higher_std_dev_gives_lower_score(self):
        """Higher std dev should give lower score."""
        score_low = consistency_score(0.1)
        score_high = consistency_score(1.0)
        assert score_low > score_high

    def test_score_is_always_positive(self):
        """Score should always be positive."""
        assert consistency_score(0.0) > 0
        assert consistency_score(1.0) > 0
        assert consistency_score(100.0) > 0

    def test_score_is_bounded_by_one(self):
        """Score should be at most 1.0."""
        assert consistency_score(0.0) <= 1.0
        assert consistency_score(0.5) <= 1.0


class TestEmbeddingProvider:
    """Tests for the EmbeddingProvider class."""

    def test_fallback_embed_returns_correct_dimension(self):
        """Fallback embed should return correct dimension."""
        provider = EmbeddingProvider()
        # Force use of fallback by not having API key
        embedding = provider._fallback_embed("Hello world", dim=384)
        assert embedding.shape == (384,)

    def test_fallback_embed_is_normalized(self):
        """Fallback embedding should be normalized."""
        provider = EmbeddingProvider()
        embedding = provider._fallback_embed("Test text with content", dim=128)
        norm = np.linalg.norm(embedding)
        # Should be normalized (or zero for empty after stop words)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == 0.0

    def test_fallback_embed_similar_texts_are_similar(self):
        """Similar texts should have similar fallback embeddings."""
        provider = EmbeddingProvider()
        e1 = provider._fallback_embed("Python programming language", dim=256)
        e2 = provider._fallback_embed("Python programming code", dim=256)
        e3 = provider._fallback_embed("Elephant zebra giraffe", dim=256)

        sim_12 = cosine_similarity(e1, e2)
        sim_13 = cosine_similarity(e1, e3)

        # Python texts should be more similar than Python and animals
        assert sim_12 > sim_13

    def test_is_configured_without_api_key(self, no_api_keys):
        """Should report not configured without API key."""
        provider = EmbeddingProvider()
        assert provider.is_configured() is False
