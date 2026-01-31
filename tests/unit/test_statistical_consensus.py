"""Tests for i2i statistical consensus mode."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from i2i.consensus import ConsensusEngine
from i2i.schema import (
    Message,
    MessageType,
    Response,
    ConfidenceLevel,
    ConsensusLevel,
    ModelStatistics,
    StatisticalConsensusResult,
)
from i2i.embeddings import EmbeddingProvider
from tests.fixtures.mock_providers import (
    MockProviderRegistry,
    MockProviderAdapter,
    create_mock_response,
)


class TestModelStatistics:
    """Tests for ModelStatistics schema."""

    def test_model_statistics_creation(self):
        """Should create ModelStatistics with defaults."""
        stats = ModelStatistics(model="test-model", n_runs=5)
        assert stats.model == "test-model"
        assert stats.n_runs == 5
        assert stats.intra_model_std_dev == 0.0
        assert stats.consistency_score == 1.0
        assert stats.outlier_indices == []
        assert stats.all_responses == []

    def test_model_statistics_with_embedding(self):
        """Should store centroid embedding."""
        stats = ModelStatistics(
            model="test-model",
            n_runs=3,
            centroid_embedding=[0.1, 0.2, 0.3],
            intra_model_std_dev=0.5,
            consistency_score=0.67,
        )
        assert stats.centroid_embedding == [0.1, 0.2, 0.3]
        assert stats.intra_model_std_dev == 0.5


class TestStatisticalConsensusResult:
    """Tests for StatisticalConsensusResult schema."""

    def test_statistical_result_creation(self):
        """Should create StatisticalConsensusResult with required fields."""
        result = StatisticalConsensusResult(
            query="What is AI?",
            models_queried=["model-a", "model-b"],
            n_runs_per_model=5,
            temperature=0.7,
            consensus_level=ConsensusLevel.HIGH,
        )
        assert result.query == "What is AI?"
        assert result.n_runs_per_model == 5
        assert result.temperature == 0.7
        assert result.total_queries == 0  # Default

    def test_statistical_result_with_statistics(self):
        """Should store per-model statistics."""
        stats_a = ModelStatistics(model="model-a", n_runs=5, consistency_score=0.9)
        stats_b = ModelStatistics(model="model-b", n_runs=5, consistency_score=0.8)

        result = StatisticalConsensusResult(
            query="Test",
            models_queried=["model-a", "model-b"],
            n_runs_per_model=5,
            temperature=0.7,
            consensus_level=ConsensusLevel.MEDIUM,
            model_statistics={"model-a": stats_a, "model-b": stats_b},
            overall_confidence=0.85,
            total_queries=10,
            total_cost_multiplier=5.0,
        )
        assert result.model_statistics["model-a"].consistency_score == 0.9
        assert result.total_cost_multiplier == 5.0


class TestComputeModelStatistics:
    """Tests for _compute_model_statistics method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        provider = MagicMock(spec=EmbeddingProvider)

        async def mock_embed_batch(texts):
            # Return simple embeddings based on text length
            return [np.array([len(t) / 100.0, 0.5, 0.3]) for t in texts]

        provider.embed_batch = AsyncMock(side_effect=mock_embed_batch)
        return provider

    async def test_computes_statistics_for_responses(self, engine, mock_embedding_provider):
        """Should compute statistics for a list of responses."""
        responses = [
            create_mock_response(model="model-a", content="Response one"),
            create_mock_response(model="model-a", content="Response two"),
            create_mock_response(model="model-a", content="Response three"),
        ]

        stats = await engine._compute_model_statistics(
            model="model-a",
            responses=responses,
            embedding_provider=mock_embedding_provider,
            outlier_threshold=2.0,
        )

        assert stats.model == "model-a"
        assert stats.n_runs == 3
        assert stats.centroid_embedding is not None
        assert stats.representative_response is not None
        assert stats.consistency_score > 0

    async def test_handles_empty_responses(self, engine, mock_embedding_provider):
        """Should handle empty response list."""
        stats = await engine._compute_model_statistics(
            model="model-a",
            responses=[],
            embedding_provider=mock_embedding_provider,
            outlier_threshold=2.0,
        )

        assert stats.n_runs == 0
        assert stats.consistency_score == 0.0

    async def test_detects_outliers(self, engine):
        """Should detect outlier responses with sufficient normal samples."""
        # Create embedding provider that returns very different embedding for one response
        # Need many normal points so outlier doesn't dominate mean/std
        provider = MagicMock(spec=EmbeddingProvider)

        async def mock_embed_with_outlier(texts):
            embeddings = []
            for i, t in enumerate(texts):
                if "outlier" in t.lower():
                    embeddings.append(np.array([100.0, 100.0, 100.0]))  # Far from others
                else:
                    # Slightly varied normal embeddings
                    embeddings.append(np.array([0.1 + i * 0.01, 0.1, 0.1]))
            return embeddings

        provider.embed_batch = AsyncMock(side_effect=mock_embed_with_outlier)

        # Need enough normal responses so outlier doesn't dominate mean/std
        responses = [
            create_mock_response(model="model-a", content="Normal response one"),
            create_mock_response(model="model-a", content="Normal response two"),
            create_mock_response(model="model-a", content="Normal response three"),
            create_mock_response(model="model-a", content="Normal response four"),
            create_mock_response(model="model-a", content="Normal response five"),
            create_mock_response(model="model-a", content="Normal response six"),
            create_mock_response(model="model-a", content="Normal response seven"),
            create_mock_response(model="model-a", content="Normal response eight"),
            create_mock_response(model="model-a", content="Normal response nine"),
            create_mock_response(model="model-a", content="OUTLIER completely different"),
        ]

        stats = await engine._compute_model_statistics(
            model="model-a",
            responses=responses,
            embedding_provider=provider,
            outlier_threshold=2.0,
        )

        assert len(stats.outlier_indices) > 0
        assert 9 in stats.outlier_indices  # The outlier at index 9


class TestStatisticalConsensusEngine:
    """Tests for the statistical consensus query method."""

    @pytest.fixture
    def consistent_registry(self):
        """Registry where each model gives very consistent responses."""
        # All responses are identical for each model
        adapter1 = MockProviderAdapter(
            "p1", ["model-a"],
            responses={"model-a": "The capital of France is Paris."}
        )
        adapter2 = MockProviderAdapter(
            "p2", ["model-b"],
            responses={"model-b": "Paris is the capital of France."}
        )
        return MockProviderRegistry({"p1": adapter1, "p2": adapter2})

    async def test_statistical_query_returns_correct_type(self, consistent_registry):
        """Statistical query should return StatisticalConsensusResult."""
        engine = ConsensusEngine(consistent_registry)

        # Mock the embedding provider
        with patch('i2i.consensus.EmbeddingProvider') as MockProvider:
            mock_provider = MagicMock()

            async def mock_embed_batch(texts):
                return [np.random.rand(3) for _ in texts]

            mock_provider.embed_batch = AsyncMock(side_effect=mock_embed_batch)
            MockProvider.return_value = mock_provider

            result = await engine.query_for_consensus_statistical(
                query="What is the capital of France?",
                models=["model-a", "model-b"],
                n_runs=3,
                temperature=0.7,
            )

        assert isinstance(result, StatisticalConsensusResult)
        assert result.n_runs_per_model == 3
        assert result.temperature == 0.7

    async def test_statistical_query_includes_model_statistics(self, consistent_registry):
        """Result should include statistics for each model."""
        engine = ConsensusEngine(consistent_registry)

        with patch('i2i.consensus.EmbeddingProvider') as MockProvider:
            mock_provider = MagicMock()

            async def mock_embed_batch(texts):
                # Return similar embeddings for consistent responses
                return [np.array([0.5, 0.5, 0.5]) + np.random.rand(3) * 0.01 for _ in texts]

            mock_provider.embed_batch = AsyncMock(side_effect=mock_embed_batch)
            MockProvider.return_value = mock_provider

            result = await engine.query_for_consensus_statistical(
                query="Test query",
                models=["model-a", "model-b"],
                n_runs=3,
            )

        # Should have statistics for both models
        assert "p1/model-a" in result.model_statistics or "model-a" in str(result.model_statistics)
        assert result.overall_confidence > 0

    async def test_total_queries_calculated_correctly(self, consistent_registry):
        """Total queries should be n_runs * num_models."""
        engine = ConsensusEngine(consistent_registry)

        with patch('i2i.consensus.EmbeddingProvider') as MockProvider:
            mock_provider = MagicMock()
            mock_provider.embed_batch = AsyncMock(
                side_effect=lambda texts: [np.random.rand(3) for _ in texts]
            )
            MockProvider.return_value = mock_provider

            result = await engine.query_for_consensus_statistical(
                query="Test",
                models=["model-a", "model-b"],
                n_runs=5,
            )

        # 2 models * 5 runs = 10 queries
        assert result.total_queries == 10
        assert result.total_cost_multiplier == 5.0


class TestStatisticalModeConfig:
    """Tests for statistical mode configuration."""

    def test_default_config_values(self):
        """Should have correct default config values."""
        from i2i.config import get_statistical_mode_config, DEFAULTS

        config = DEFAULTS["statistical_mode"]
        assert config["enabled"] is False
        assert config["n_runs"] == 5
        assert config["temperature"] == 0.7
        assert config["outlier_threshold"] == 2.0

    def test_is_statistical_mode_enabled_default(self, clean_env):
        """Statistical mode should be disabled by default."""
        from i2i.config import is_statistical_mode_enabled, reset_config

        reset_config()
        assert is_statistical_mode_enabled() is False

    def test_statistical_mode_env_override(self, clean_env):
        """Environment variable should override statistical mode."""
        import os
        from i2i.config import reset_config, get_config

        os.environ["I2I_STATISTICAL_MODE"] = "true"
        reset_config()

        config = get_config()
        assert config.get("statistical_mode.enabled") is True

        # Clean up
        del os.environ["I2I_STATISTICAL_MODE"]

    def test_n_runs_env_override(self, clean_env):
        """Environment variable should override n_runs."""
        import os
        from i2i.config import reset_config, get_statistical_n_runs

        os.environ["I2I_STATISTICAL_N_RUNS"] = "10"
        reset_config()

        n_runs = get_statistical_n_runs()
        assert n_runs == 10

        # Clean up
        del os.environ["I2I_STATISTICAL_N_RUNS"]


class TestAnalyzeConsensusWithEmbeddings:
    """Tests for embedding-based consensus analysis."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    async def test_high_similarity_embeddings_give_high_consensus(self, engine):
        """Very similar embeddings should give HIGH consensus."""
        responses = [
            create_mock_response(model="model-a", content="Similar response A"),
            create_mock_response(model="model-b", content="Similar response B"),
        ]

        provider = MagicMock(spec=EmbeddingProvider)
        # Return almost identical embeddings
        provider.embed_batch = AsyncMock(
            return_value=[np.array([1.0, 0.0, 0.0]), np.array([0.99, 0.01, 0.0])]
        )

        level, matrix = await engine._analyze_consensus_with_embeddings(responses, provider)
        assert level == ConsensusLevel.HIGH

    async def test_low_similarity_embeddings_give_low_consensus(self, engine):
        """Very different embeddings should give LOW or NONE consensus."""
        responses = [
            create_mock_response(model="model-a", content="Response A"),
            create_mock_response(model="model-b", content="Response B"),
        ]

        provider = MagicMock(spec=EmbeddingProvider)
        # Return orthogonal embeddings
        provider.embed_batch = AsyncMock(
            return_value=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        )

        level, matrix = await engine._analyze_consensus_with_embeddings(responses, provider)
        assert level in [ConsensusLevel.LOW, ConsensusLevel.NONE]

    async def test_agreement_matrix_contains_similarities(self, engine):
        """Agreement matrix should contain pairwise similarities."""
        responses = [
            create_mock_response(model="model-a", content="A"),
            create_mock_response(model="model-b", content="B"),
        ]

        provider = MagicMock(spec=EmbeddingProvider)
        provider.embed_batch = AsyncMock(
            return_value=[np.array([1.0, 0.0]), np.array([0.8, 0.6])]
        )

        _, matrix = await engine._analyze_consensus_with_embeddings(responses, provider)

        assert "model-a" in matrix
        assert "model-b" in matrix
        # Self-similarity should be 1.0
        assert matrix["model-a"]["model-a"] == 1.0
        assert matrix["model-b"]["model-b"] == 1.0
