# InsightSpike Extended Analysis

*Generated: 2026-01-19 20:49:40*

## Raw Data

```json
{
  "timestamp": "2026-01-19T20:49:40.303025",
  "total_methods": 1105,
  "stats": {
    "total": 1105,
    "by_type": {
      "class": 814,
      "function": 291
    },
    "modules": 224
  },
  "module_structure": {
    "module_count": 224,
    "modules": {
      "insightspike": {
        "total": 22,
        "classes": 13,
        "functions": 9,
        "class_names": [
          "About",
          "AgentConfigBuilder",
          "CycleResult",
          "EnvironmentInterface",
          "ErrorMonitor",
          "GenericInsightSpikeAgent",
          "InsightMoment",
          "InsightSpikeAgentFactory",
          "L2MemoryManager",
          "L3GraphReasoner",
          "MainAgent",
          "StandaloneL3GraphReasoner",
          "TaskType"
        ],
        "function_names": [
          "about",
          "analyze_documents_simple",
          "create_agent",
          "create_configured_maze_agent",
          "create_maze_agent",
          "create_standalone_reasoner",
          "get_config",
          "get_llm_provider",
          "quick_demo"
        ]
      },
      "insightspike.__main__": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "run_app"
        ]
      },
      "insightspike.adaptive": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "AdaptiveProcessor",
          "ExplorationLoop",
          "ExplorationParams",
          "ExplorationResult",
          "ExplorationStrategy",
          "PatternLearner",
          "TopKCalculator"
        ],
        "function_names": []
      },
      "insightspike.adaptive.calculators.adaptive_topk": {
        "total": 4,
        "classes": 3,
        "functions": 1,
        "class_names": [
          "AdaptiveTopKCalculator",
          "AdaptiveTopKConfig",
          "TopKCalculator"
        ],
        "function_names": [
          "estimate_chain_reaction_potential"
        ]
      },
      "insightspike.adaptive.calculators.simple_topk": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "SimpleTopKCalculator"
        ],
        "function_names": []
      },
      "insightspike.adaptive.core.adaptive_processor": {
        "total": 8,
        "classes": 8,
        "functions": 0,
        "class_names": [
          "AdaptiveProcessor",
          "DataStore",
          "ExplorationLoop",
          "ExplorationParams",
          "ExplorationResult",
          "ExplorationStrategy",
          "PatternLearner",
          "TopKCalculator"
        ],
        "function_names": []
      },
      "insightspike.adaptive.core.exploration_loop": {
        "total": 5,
        "classes": 4,
        "functions": 1,
        "class_names": [
          "ExplorationLoop",
          "ExplorationParams",
          "ExplorationResult",
          "NormalizedConfig"
        ],
        "function_names": [
          "get_config"
        ]
      },
      "insightspike.adaptive.core.exploration_loop_fixed": {
        "total": 5,
        "classes": 4,
        "functions": 1,
        "class_names": [
          "ExplorationLoopFixed",
          "ExplorationParams",
          "ExplorationResult",
          "NormalizedConfig"
        ],
        "function_names": [
          "get_config"
        ]
      },
      "insightspike.adaptive.core.interfaces": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "ExplorationParams",
          "ExplorationResult",
          "ExplorationStrategy",
          "PatternLearner",
          "TopKCalculator"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "AlternatingStrategy",
          "ExpandingStrategy",
          "ExponentialStrategy",
          "NarrowingStrategy"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies.alternating": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "AlternatingStrategy",
          "BaseStrategy",
          "ExplorationParams",
          "ExplorationResult"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies.base": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "AdaptiveTopKCalculator",
          "BaseStrategy",
          "ExplorationParams",
          "ExplorationResult",
          "ExplorationStrategy"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies.expanding": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "BaseStrategy",
          "ExpandingStrategy",
          "ExplorationParams",
          "ExplorationResult"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies.exponential_strategy": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "BaseStrategy",
          "ExponentialStrategy"
        ],
        "function_names": []
      },
      "insightspike.adaptive.strategies.narrowing": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "BaseStrategy",
          "ExplorationParams",
          "ExplorationResult",
          "NarrowingStrategy"
        ],
        "function_names": []
      },
      "insightspike.algorithms": {
        "total": 22,
        "classes": 9,
        "functions": 13,
        "class_names": [
          "ContentStructureSeparation",
          "EntropyCalculator",
          "EntropyMethod",
          "EntropyResult",
          "GEDResult",
          "GraphEditDistance",
          "IGResult",
          "InformationGain",
          "OptimizationLevel"
        ],
        "function_names": [
          "clustering_coefficient_entropy",
          "compute_delta_ged",
          "compute_delta_ig",
          "compute_graph_edit_distance",
          "compute_information_gain",
          "compute_shannon_entropy",
          "create_default_ged_calculator",
          "create_default_ig_calculator",
          "degree_distribution_entropy",
          "get_algorithm_info",
          "path_length_entropy",
          "structural_entropy",
          "von_neumann_entropy"
        ]
      },
      "insightspike.algorithms.core": {
        "total": 5,
        "classes": 0,
        "functions": 5,
        "class_names": [],
        "function_names": [
          "entropy_ig",
          "graph_efficiency",
          "local_entropies",
          "normalized_ged",
          "spectral_score"
        ]
      },
      "insightspike.algorithms.core.metrics": {
        "total": 5,
        "classes": 0,
        "functions": 5,
        "class_names": [],
        "function_names": [
          "entropy_ig",
          "graph_efficiency",
          "local_entropies",
          "normalized_ged",
          "spectral_score"
        ]
      },
      "insightspike.algorithms.entropy_calculator": {
        "total": 9,
        "classes": 5,
        "functions": 4,
        "class_names": [
          "ContentStructureSeparation",
          "EntropyCalculator",
          "EntropyMethod",
          "EntropyResult",
          "InformationGain"
        ],
        "function_names": [
          "calc_structural_entropy",
          "clustering_coefficient_entropy",
          "degree_distribution_entropy",
          "von_neumann_entropy"
        ]
      },
      "insightspike.algorithms.gating": {
        "total": 2,
        "classes": 1,
        "functions": 1,
        "class_names": [
          "GateDecision"
        ],
        "function_names": [
          "decide_gates"
        ]
      },
      "insightspike.algorithms.gedig.ab_writer_helper": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "create_csv_writer"
        ]
      },
      "insightspike.algorithms.gedig.selector": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "TwoThresholdCandidateSelector",
          "TwoThresholdSelection"
        ],
        "function_names": [
          "compute_gedig"
        ]
      },
      "insightspike.algorithms.gedig_ab_logger": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "GeDIGABLogger",
          "GeDIGResult"
        ],
        "function_names": []
      },
      "insightspike.algorithms.gedig_analysis": {
        "total": 2,
        "classes": 1,
        "functions": 1,
        "class_names": [
          "DivergenceStats"
        ],
        "function_names": [
          "analyze_divergence"
        ]
      },
      "insightspike.algorithms.gedig_calculator": {
        "total": 4,
        "classes": 3,
        "functions": 1,
        "class_names": [
          "GeDIGCalculator",
          "GeDIGCore",
          "GeDIGResult"
        ],
        "function_names": [
          "calculate_gedig"
        ]
      },
      "insightspike.algorithms.gedig_core": {
        "total": 13,
        "classes": 9,
        "functions": 4,
        "class_names": [
          "GeDIGCore",
          "GeDIGLogger",
          "GeDIGMonitor",
          "GeDIGPresets",
          "GeDIGResult",
          "HopResult",
          "LinksetMetrics",
          "ProcessingMode",
          "SpikeDetectionMode"
        ],
        "function_names": [
          "calculate_gedig",
          "delta_ged",
          "delta_ig",
          "detect_insight_spike"
        ]
      },
      "insightspike.algorithms.gedig_factory": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "GeDIGCore",
          "GeDIGFactory"
        ],
        "function_names": [
          "dual_evaluate"
        ]
      },
      "insightspike.algorithms.gedig_pure": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "PureGeDIGCalculator",
          "PureGeDIGResult"
        ],
        "function_names": []
      },
      "insightspike.algorithms.gedig_utils": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "GEDCalculationUtils",
          "GeDIGComputationUtils",
          "GraphValidationUtils",
          "IGCalculationUtils"
        ],
        "function_names": []
      },
      "insightspike.algorithms.gedig_wake_mode": {
        "total": 6,
        "classes": 5,
        "functions": 1,
        "class_names": [
          "GeDIGCore",
          "GeDIGResult",
          "ProcessingMode",
          "WakeModeGeDIG",
          "WakeModeResult"
        ],
        "function_names": [
          "calculate_wake_mode_gedig"
        ]
      },
      "insightspike.algorithms.geometric_probabilistic_graph": {
        "total": 5,
        "classes": 3,
        "functions": 2,
        "class_names": [
          "GeometricProbabilisticGraph",
          "GraphAttentionProbability",
          "ThreeSpaceMetrics"
        ],
        "function_names": [
          "calculate_three_space_entropy",
          "calculate_three_space_information_gain"
        ]
      },
      "insightspike.algorithms.graph_edit_distance": {
        "total": 7,
        "classes": 3,
        "functions": 4,
        "class_names": [
          "GEDResult",
          "GraphEditDistance",
          "OptimizationLevel"
        ],
        "function_names": [
          "compute_delta_ged",
          "compute_graph_edit_distance",
          "get_global_ged_calculator",
          "reset_ged_state"
        ]
      },
      "insightspike.algorithms.graph_importance": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "DynamicImportanceTracker",
          "GraphImportanceCalculator"
        ],
        "function_names": []
      },
      "insightspike.algorithms.graph_structure_analyzer": {
        "total": 2,
        "classes": 1,
        "functions": 1,
        "class_names": [
          "GraphStructureAnalyzer"
        ],
        "function_names": [
          "analyze_graph_structure"
        ]
      },
      "insightspike.algorithms.improved_similarity_entropy": {
        "total": 8,
        "classes": 1,
        "functions": 7,
        "class_names": [
          "NormalizationMethod"
        ],
        "function_names": [
          "calculate_entropy_change",
          "calculate_information_gain",
          "calculate_similarity_entropy",
          "exponential_normalization",
          "linear_normalization",
          "piecewise_normalization",
          "sigmoid_normalization"
        ]
      },
      "insightspike.algorithms.information_gain": {
        "total": 10,
        "classes": 4,
        "functions": 6,
        "class_names": [
          "EntropyMethod",
          "IGResult",
          "ImprovedEntropyMethods",
          "InformationGain"
        ],
        "function_names": [
          "calc_structural_entropy",
          "compute_delta_ig",
          "compute_information_gain",
          "compute_shannon_entropy",
          "degree_distribution_entropy",
          "von_neumann_entropy"
        ]
      },
      "insightspike.algorithms.isomorphism_discovery": {
        "total": 7,
        "classes": 5,
        "functions": 2,
        "class_names": [
          "EditOperation",
          "EditType",
          "IsomorphismDiscovery",
          "PartialMapping",
          "Transform"
        ],
        "function_names": [
          "create_test_graphs",
          "discover_insight"
        ]
      },
      "insightspike.algorithms.linkset_adapter": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "build_linkset_info"
        ]
      },
      "insightspike.algorithms.local_information_gain_v2": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "LocalIGResult",
          "LocalInformationGainV2"
        ],
        "function_names": [
          "compute_local_ig"
        ]
      },
      "insightspike.algorithms.metrics_selector": {
        "total": 12,
        "classes": 3,
        "functions": 9,
        "class_names": [
          "GraphEditDistance",
          "InformationGain",
          "MetricsSelector"
        ],
        "function_names": [
          "advanced_delta_ged",
          "advanced_delta_ig",
          "delta_ged",
          "delta_ged_pyg",
          "delta_ig",
          "delta_ig_pyg",
          "get_metrics_selector",
          "simple_delta_ged",
          "simple_delta_ig"
        ]
      },
      "insightspike.algorithms.proper_delta_ged": {
        "total": 3,
        "classes": 1,
        "functions": 2,
        "class_names": [
          "ProperDeltaGED"
        ],
        "function_names": [
          "proper_delta_ged",
          "reset_ged_state"
        ]
      },
      "insightspike.algorithms.proper_entropy": {
        "total": 2,
        "classes": 0,
        "functions": 2,
        "class_names": [],
        "function_names": [
          "calculate_information_gain",
          "calculate_vector_entropy"
        ]
      },
      "insightspike.algorithms.pyg_adapter": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "PyGAdapter",
          "PyGGraphEditDistance"
        ],
        "function_names": []
      },
      "insightspike.algorithms.query_type_processor": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "QueryContext",
          "QueryType",
          "QueryTypeProcessor",
          "WakeModeGeDIG",
          "WakeModeResult"
        ],
        "function_names": []
      },
      "insightspike.algorithms.rehydration": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "RehydrationStats",
          "Rehydrator"
        ],
        "function_names": []
      },
      "insightspike.algorithms.similarity_entropy": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "calculate_similarity_entropy"
        ]
      },
      "insightspike.algorithms.sp_distcache": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "DistanceCache",
          "PairSet"
        ],
        "function_names": []
      },
      "insightspike.algorithms.structural_entropy": {
        "total": 5,
        "classes": 0,
        "functions": 5,
        "class_names": [],
        "function_names": [
          "clustering_coefficient_entropy",
          "degree_distribution_entropy",
          "path_length_entropy",
          "structural_entropy",
          "von_neumann_entropy"
        ]
      },
      "insightspike.algorithms.structural_similarity": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "SimilarityResult",
          "StructuralSimilarityEvaluator"
        ],
        "function_names": [
          "compute_structural_similarity"
        ]
      },
      "insightspike.algorithms.structure_enricher": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "StructuralSimilarityConfig",
          "StructuralSimilarityEvaluator",
          "StructureEnricher"
        ],
        "function_names": []
      },
      "insightspike.cli.commands": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "discover_command"
        ]
      },
      "insightspike.cli.commands.bridge": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "ConceptBridge",
          "SimpleRAGGraph"
        ],
        "function_names": [
          "bridge_command",
          "visualize_bridge_path"
        ]
      },
      "insightspike.cli.commands.discover": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "InsightDiscovery",
          "MainAgent"
        ],
        "function_names": [
          "discover_command",
          "get_logger"
        ]
      },
      "insightspike.cli.commands.graph": {
        "total": 11,
        "classes": 4,
        "functions": 7,
        "class_names": [
          "DataStoreConfig",
          "DataStoreFactory",
          "GraphAnalyzer",
          "SimpleRAGGraph"
        ],
        "function_names": [
          "analyze_command",
          "graph_command",
          "load_config",
          "resolve_project_relative",
          "sleep_gc_command",
          "visualize_command",
          "visualize_graph_metrics"
        ]
      },
      "insightspike.cli.legacy": {
        "total": 10,
        "classes": 2,
        "functions": 8,
        "class_names": [
          "InsightFactRegistry",
          "MainAgent"
        ],
        "function_names": [
          "ask",
          "config_info",
          "get_config",
          "load_corpus",
          "load_documents",
          "main",
          "main_callback",
          "stats"
        ]
      },
      "insightspike.cli.simple_cli": {
        "total": 2,
        "classes": 0,
        "functions": 2,
        "class_names": [],
        "function_names": [
          "test",
          "version"
        ]
      },
      "insightspike.cli.spike": {
        "total": 28,
        "classes": 7,
        "functions": 21,
        "class_names": [
          "CLIState",
          "ConfigLoader",
          "ConfigPresets",
          "DependencyFactory",
          "InsightSpikeConfig",
          "InsightSpikeError",
          "MainAgent"
        ],
        "function_names": [
          "analyze_command",
          "bridge_command",
          "config",
          "demo",
          "discover_command",
          "embed",
          "experiment",
          "get_config",
          "get_logger",
          "insights",
          "insights_search",
          "interactive",
          "load_config",
          "main",
          "query",
          "run_cli",
          "show_stats",
          "sleep_gc_command",
          "stats",
          "version",
          "visualize_command"
        ]
      },
      "insightspike.config": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "ConfigPresets",
          "InsightSpikeConfig"
        ],
        "function_names": [
          "get_config",
          "load_config"
        ]
      },
      "insightspike.config.compat": {
        "total": 5,
        "classes": 2,
        "functions": 3,
        "class_names": [
          "ConfigNormalizer",
          "InsightSpikeConfig"
        ],
        "function_names": [
          "detect_config_type",
          "is_pydantic_config",
          "normalize"
        ]
      },
      "insightspike.config.constants": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "DataType",
          "Defaults",
          "FileFormat"
        ],
        "function_names": []
      },
      "insightspike.config.index_config": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "IndexFeatureFlags",
          "IntegratedIndexConfig"
        ],
        "function_names": []
      },
      "insightspike.config.legacy_adapter": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "ConfigNormalizer",
          "InsightSpikeConfig",
          "LegacyConfigAdapter"
        ],
        "function_names": []
      },
      "insightspike.config.loader": {
        "total": 5,
        "classes": 3,
        "functions": 2,
        "class_names": [
          "ConfigLoader",
          "ConfigPresets",
          "InsightSpikeConfig"
        ],
        "function_names": [
          "get_config",
          "load_config"
        ]
      },
      "insightspike.config.message_passing_config": {
        "total": 4,
        "classes": 1,
        "functions": 3,
        "class_names": [
          "MessagePassingConfig"
        ],
        "function_names": [
          "get_default_message_passing_config",
          "get_performance_optimized_config",
          "get_quality_optimized_config"
        ]
      },
      "insightspike.config.models": {
        "total": 24,
        "classes": 24,
        "functions": 0,
        "class_names": [
          "DataStoreConfig",
          "DefaultsAppliedMixin",
          "EmbeddingConfig",
          "GraphConfig",
          "HybridWeightsConfig",
          "InsightSpikeConfig",
          "LLMConfig",
          "LoggingConfig",
          "MazeConfig",
          "MazeExperimentConfig",
          "MazeNavigatorConfig",
          "MemoryConfig",
          "MetricsConfig",
          "MonitoringConfig",
          "MultihopConfig",
          "OutputConfig",
          "PathsConfig",
          "PerformanceConfig",
          "ProcessingConfig",
          "ReasoningConfig",
          "SpectralEvaluationConfig",
          "StructuralSimilarityConfig",
          "VectorSearchConfig",
          "WakeSleepConfig"
        ],
        "function_names": []
      },
      "insightspike.config.normalized": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "InsightSpikeConfig",
          "NormalizedConfig"
        ],
        "function_names": []
      },
      "insightspike.config.normalizer": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "ConfigNormalizer",
          "EmbeddingConfig",
          "GraphConfig",
          "InsightSpikeConfig",
          "LLMConfig",
          "MemoryConfig",
          "ProcessingConfig"
        ],
        "function_names": []
      },
      "insightspike.config.presets": {
        "total": 8,
        "classes": 8,
        "functions": 0,
        "class_names": [
          "ConfigPresets",
          "EmbeddingConfig",
          "GraphConfig",
          "InsightSpikeConfig",
          "LLMConfig",
          "LoggingConfig",
          "MemoryConfig",
          "MonitoringConfig"
        ],
        "function_names": []
      },
      "insightspike.config.summary": {
        "total": 2,
        "classes": 0,
        "functions": 2,
        "class_names": [],
        "function_names": [
          "summarize_config",
          "summarize_memory_config"
        ]
      },
      "insightspike.config.vector_weights": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "VectorWeightConfig"
        ],
        "function_names": []
      },
      "insightspike.config.wake_sleep_config": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "SleepModeConfig",
          "SphereSearchConfig",
          "WakeModeConfig",
          "WakeSleepConfig"
        ],
        "function_names": []
      },
      "insightspike.core": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "CycleResult",
          "Episode",
          "GraphMetrics",
          "MemoryController",
          "ResponseGenerator"
        ],
        "function_names": []
      },
      "insightspike.core.base": {
        "total": 23,
        "classes": 23,
        "functions": 0,
        "class_names": [
          "ActionSpace",
          "AgentInterface",
          "EnvironmentInterface",
          "EnvironmentState",
          "GenericAgentInterface",
          "InsightDetectorInterface",
          "InsightMoment",
          "L1ErrorMonitorInterface",
          "L2MemoryInterface",
          "L3GraphReasonerInterface",
          "L4LLMInterface",
          "LayerInput",
          "LayerInterface",
          "LayerOutput",
          "MazeEnvironmentAdapter",
          "MazeInsightDetector",
          "MazeRewardNormalizer",
          "MazeStateEncoder",
          "MemoryManagerInterface",
          "ReasonerInterface",
          "RewardNormalizer",
          "StateEncoder",
          "TaskType"
        ],
        "function_names": []
      },
      "insightspike.core.base.async_datastore": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "AsyncDataStore",
          "DataStore"
        ],
        "function_names": []
      },
      "insightspike.core.base.datastore": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "DataStore",
          "VectorIndex"
        ],
        "function_names": []
      },
      "insightspike.core.base.generic_interfaces": {
        "total": 11,
        "classes": 11,
        "functions": 0,
        "class_names": [
          "ActionSpace",
          "EnvironmentInterface",
          "EnvironmentState",
          "GenericAgentInterface",
          "InsightDetectorInterface",
          "InsightMoment",
          "MemoryManagerInterface",
          "ReasonerInterface",
          "RewardNormalizer",
          "StateEncoder",
          "TaskType"
        ],
        "function_names": []
      },
      "insightspike.core.base.layer4_interface": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "L4Interface",
          "L4_1Interface"
        ],
        "function_names": []
      },
      "insightspike.core.base.maze_implementation": {
        "total": 12,
        "classes": 12,
        "functions": 0,
        "class_names": [
          "ActionSpace",
          "EnvironmentInterface",
          "EnvironmentState",
          "InsightDetectorInterface",
          "InsightMoment",
          "MazeEnvironmentAdapter",
          "MazeInsightDetector",
          "MazeRewardNormalizer",
          "MazeStateEncoder",
          "RewardNormalizer",
          "StateEncoder",
          "TaskType"
        ],
        "function_names": []
      },
      "insightspike.core.enhanced_episode": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "EdgeInfo",
          "EnhancedEpisode",
          "Episode",
          "MemoryStatistics"
        ],
        "function_names": []
      },
      "insightspike.core.episode": {
        "total": 2,
        "classes": 1,
        "functions": 1,
        "class_names": [
          "Episode"
        ],
        "function_names": [
          "create_episode"
        ]
      },
      "insightspike.core.error_handler": {
        "total": 11,
        "classes": 6,
        "functions": 5,
        "class_names": [
          "ConfigurationError",
          "InitializationError",
          "InsightSpikeError",
          "InsightSpikeLogger",
          "ModelNotFoundError",
          "ProcessingError"
        ],
        "function_names": [
          "get_logger",
          "handle_error",
          "setup_debug_mode",
          "validate_config",
          "with_error_handling"
        ]
      },
      "insightspike.core.exceptions": {
        "total": 22,
        "classes": 22,
        "functions": 0,
        "class_names": [
          "AgentError",
          "AgentInitializationError",
          "AgentProcessingError",
          "ConfigNotFoundError",
          "ConfigurationError",
          "DataStoreError",
          "DataStoreLoadError",
          "DataStoreNotFoundError",
          "DataStorePermissionError",
          "DataStoreSaveError",
          "GraphAnalysisError",
          "GraphBuildError",
          "GraphError",
          "InsightSpikeException",
          "InvalidConfigError",
          "LLMConnectionError",
          "LLMError",
          "LLMGenerationError",
          "LLMTokenLimitError",
          "MemoryCapacityError",
          "MemoryError",
          "MemorySearchError"
        ],
        "function_names": []
      },
      "insightspike.core.memory_controller": {
        "total": 6,
        "classes": 6,
        "functions": 0,
        "class_names": [
          "Episode",
          "IDataStore",
          "IEmbedder",
          "IMemoryManager",
          "MemoryController",
          "PyGGraphBuilder"
        ],
        "function_names": []
      },
      "insightspike.core.reasoning_engine": {
        "total": 6,
        "classes": 6,
        "functions": 0,
        "class_names": [
          "CycleResult",
          "ILLMProvider",
          "IMemorySearch",
          "L3GraphReasoner",
          "L4LLMInterface",
          "ReasoningEngine"
        ],
        "function_names": []
      },
      "insightspike.core.response_generator": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "CycleResult",
          "ILLMProvider",
          "ResponseGenerator",
          "ResponseTemplate"
        ],
        "function_names": []
      },
      "insightspike.core.structures": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "CycleResult",
          "GraphMetrics"
        ],
        "function_names": []
      },
      "insightspike.core.vector_integrator": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "VectorIntegrator"
        ],
        "function_names": []
      },
      "insightspike.core.weight_vector_manager": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "VectorWeightConfig",
          "WeightVectorManager"
        ],
        "function_names": []
      },
      "insightspike.decoder": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "VectorDecoder"
        ],
        "function_names": []
      },
      "insightspike.decoder.vector_decoder": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "VectorDecoder"
        ],
        "function_names": []
      },
      "insightspike.detection": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "EurekaDetector",
          "InsightFactRegistry"
        ],
        "function_names": [
          "detect_eureka_spike"
        ]
      },
      "insightspike.detection.eureka_spike": {
        "total": 3,
        "classes": 1,
        "functions": 2,
        "class_names": [
          "EurekaDetector"
        ],
        "function_names": [
          "detect_eureka_spike",
          "get_config"
        ]
      },
      "insightspike.detection.insight_registry": {
        "total": 5,
        "classes": 3,
        "functions": 2,
        "class_names": [
          "GraphOptimizationResult",
          "InsightFact",
          "InsightFactRegistry"
        ],
        "function_names": [
          "get_insight_registry",
          "shutdown_insight_registry"
        ]
      },
      "insightspike.di": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "DIContainer",
          "DataStoreProvider",
          "EmbedderProvider",
          "GraphBuilderProvider",
          "LLMProviderFactory",
          "MemoryManagerProvider",
          "ServiceProvider"
        ],
        "function_names": []
      },
      "insightspike.di.container": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "DIContainer",
          "FactoryProvider",
          "ServiceProvider",
          "SingletonProvider"
        ],
        "function_names": []
      },
      "insightspike.di.providers": {
        "total": 16,
        "classes": 16,
        "functions": 0,
        "class_names": [
          "DataStoreFactory",
          "DataStoreProvider",
          "EmbedderProvider",
          "EmbeddingManager",
          "GraphBuilderProvider",
          "IDataStore",
          "IEmbedder",
          "IGraphBuilder",
          "ILLMProvider",
          "IMemoryManager",
          "InsightSpikeConfig",
          "L2MemoryManager",
          "LLMProviderFactory",
          "LLMProviderRegistry",
          "MemoryManagerProvider",
          "PyGGraphBuilder"
        ],
        "function_names": []
      },
      "insightspike.environments.complex_maze": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "ComplexMazeGenerator"
        ],
        "function_names": []
      },
      "insightspike.environments.maze": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "CellType",
          "ComplexMazeGenerator",
          "MazeObservation",
          "ProperMazeGenerator",
          "SimpleMaze"
        ],
        "function_names": []
      },
      "insightspike.environments.proper_maze_generator": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "ProperMazeGenerator"
        ],
        "function_names": []
      },
      "insightspike.features.graph_reasoning": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "GraphAnalyzer",
          "RewardCalculator"
        ],
        "function_names": []
      },
      "insightspike.features.graph_reasoning.graph_analyzer": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "GraphAnalyzer"
        ],
        "function_names": []
      },
      "insightspike.features.graph_reasoning.reward_calculator": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "RewardCalculator"
        ],
        "function_names": []
      },
      "insightspike.features.query_transformation": {
        "total": 12,
        "classes": 12,
        "functions": 0,
        "class_names": [
          "AdaptiveExplorer",
          "EnhancedQueryTransformer",
          "EvolutionPattern",
          "EvolutionTracker",
          "MultiHopGNN",
          "PatternDatabase",
          "QueryBranch",
          "QueryState",
          "QueryTransformationHistory",
          "QueryTransformer",
          "QueryTypeClassifier",
          "TrajectoryAnalyzer"
        ],
        "function_names": []
      },
      "insightspike.features.query_transformation.enhanced_query_transformer": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "AdaptiveExplorer",
          "EnhancedQueryTransformer",
          "MultiHopGNN",
          "QueryBranch",
          "QueryGraphGNN",
          "QueryState",
          "QueryTransformer"
        ],
        "function_names": []
      },
      "insightspike.features.query_transformation.evolution_tracker": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "EvolutionPattern",
          "EvolutionTracker",
          "PatternDatabase",
          "QueryState",
          "QueryTransformationHistory",
          "QueryTypeClassifier",
          "TrajectoryAnalyzer"
        ],
        "function_names": []
      },
      "insightspike.features.query_transformation.query_state": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "QueryState",
          "QueryTransformationHistory"
        ],
        "function_names": []
      },
      "insightspike.features.query_transformation.query_transformer": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "GraphEditDistance",
          "QueryGraphGNN",
          "QueryState",
          "QueryTransformationHistory",
          "QueryTransformer"
        ],
        "function_names": []
      },
      "insightspike.graph.construction": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "GraphBuilder"
        ],
        "function_names": []
      },
      "insightspike.graph.edge_reevaluator": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "EdgeReevaluator"
        ],
        "function_names": []
      },
      "insightspike.graph.message_passing": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "MessagePassing"
        ],
        "function_names": []
      },
      "insightspike.graph.message_passing_optimized": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "OptimizedMessagePassing"
        ],
        "function_names": []
      },
      "insightspike.graph.type_adapter": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "GraphTypeAdapter"
        ],
        "function_names": []
      },
      "insightspike.implementations.agents": {
        "total": 10,
        "classes": 8,
        "functions": 2,
        "class_names": [
          "AgentConfig",
          "AgentConfigBuilder",
          "AgentMode",
          "ConfigurableAgent",
          "CycleResult",
          "GenericInsightSpikeAgent",
          "InsightSpikeAgentFactory",
          "MainAgent"
        ],
        "function_names": [
          "create_maze_agent",
          "create_qa_agent"
        ]
      },
      "insightspike.implementations.agents.agent_factory": {
        "total": 11,
        "classes": 8,
        "functions": 3,
        "class_names": [
          "AgentConfigBuilder",
          "GenericInsightSpikeAgent",
          "InsightSpikeAgentFactory",
          "MazeEnvironmentAdapter",
          "MazeInsightDetector",
          "MazeRewardNormalizer",
          "MazeStateEncoder",
          "TaskType"
        ],
        "function_names": [
          "create_configured_maze_agent",
          "create_maze_agent",
          "create_qa_agent"
        ]
      },
      "insightspike.implementations.agents.configurable_agent": {
        "total": 8,
        "classes": 7,
        "functions": 1,
        "class_names": [
          "AgentConfig",
          "AgentMode",
          "ConfigurableAgent",
          "MainAgent",
          "QueryCache",
          "UnifiedCycleResult",
          "UnifiedMainAgent"
        ],
        "function_names": [
          "create_agent"
        ]
      },
      "insightspike.implementations.agents.datastore_agent": {
        "total": 9,
        "classes": 9,
        "functions": 0,
        "class_names": [
          "DataStore",
          "DataStoreMainAgent",
          "EmbeddingManager",
          "Episode",
          "EurekaDetector",
          "InsightSpikeConfig",
          "L2WorkingMemoryManager",
          "L4LLMInterface",
          "WorkingMemoryConfig"
        ],
        "function_names": []
      },
      "insightspike.implementations.agents.decision_controller": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "Decision",
          "DecisionController"
        ],
        "function_names": []
      },
      "insightspike.implementations.agents.generic_agent": {
        "total": 13,
        "classes": 13,
        "functions": 0,
        "class_names": [
          "EnvironmentInterface",
          "EnvironmentState",
          "GenericAgentInterface",
          "GenericInsightSpikeAgent",
          "GenericMemoryManager",
          "GenericReasoner",
          "InsightDetectorInterface",
          "InsightMoment",
          "MemoryManagerInterface",
          "ReasonerInterface",
          "RewardNormalizer",
          "StateEncoder",
          "TaskType"
        ],
        "function_names": []
      },
      "insightspike.implementations.agents.main_agent": {
        "total": 20,
        "classes": 15,
        "functions": 5,
        "class_names": [
          "AgentConfig",
          "AgentMode",
          "ConfigurableAgent",
          "CycleResult",
          "DataStore",
          "Episode",
          "ErrorMonitor",
          "GeDIGABLogger",
          "GeDIGFallbackTracker",
          "GraphMemorySearch",
          "InsightSpikeConfig",
          "L3GraphReasoner",
          "MainAgent",
          "Memory",
          "TwoThresholdCandidateSelector"
        ],
        "function_names": [
          "cycle",
          "get_insight_registry",
          "get_llm_provider",
          "safe_attr",
          "safe_has"
        ]
      },
      "insightspike.implementations.agents.slim_main_agent": {
        "total": 15,
        "classes": 11,
        "functions": 4,
        "class_names": [
          "CycleResult",
          "DataStore",
          "Episode",
          "ErrorMonitor",
          "FallbackReason",
          "GraphMemorySearch",
          "L3GraphReasoner",
          "Memory",
          "SlimMainAgent",
          "SpikeDecisionMode",
          "SpikePipeline"
        ],
        "function_names": [
          "create_slim_agent",
          "execute_fallback",
          "get_fallback_registry",
          "get_insight_registry"
        ]
      },
      "insightspike.implementations.datastore": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "DataStoreFactory",
          "FileSystemDataStore",
          "InMemoryDataStore"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.adapters": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "DataStore",
          "Episode",
          "L2MemoryAdapter",
          "L3GraphAdapter"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.configurable_vector_index": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "ConfigurableVectorIndex",
          "VectorIndex",
          "VectorIndexFactory"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.enhanced_filesystem_store": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "BackwardCompatibleWrapper",
          "DataStore",
          "EnhancedFileSystemDataStore",
          "FileSystemDataStore",
          "IntegratedVectorGraphIndex"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.factory": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "DataStore",
          "DataStoreFactory",
          "FileSystemDataStore",
          "InMemoryDataStore",
          "SQLiteDataStore"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.filesystem_store": {
        "total": 11,
        "classes": 10,
        "functions": 1,
        "class_names": [
          "DataStore",
          "DataStoreLoadError",
          "DataStoreNotFoundError",
          "DataStorePermissionError",
          "DataStoreSaveError",
          "DataType",
          "FAISSVectorIndex",
          "FileFormat",
          "FileSystemDataStore",
          "VectorIndex"
        ],
        "function_names": [
          "resolve_project_relative"
        ]
      },
      "insightspike.implementations.datastore.memory_store": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "DataStore",
          "InMemoryDataStore"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.sqlite_store": {
        "total": 6,
        "classes": 6,
        "functions": 0,
        "class_names": [
          "AsyncDataStore",
          "ConfigurableVectorIndex",
          "FAISSVectorIndex",
          "SQLiteDataStore",
          "VectorIndex",
          "VectorIndexFactory"
        ],
        "function_names": []
      },
      "insightspike.implementations.datastore.sqlite_store_graph": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "SQLiteDataStoreGraphMixin"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers": {
        "total": 11,
        "classes": 10,
        "functions": 1,
        "class_names": [
          "CompatibleL2MemoryManager",
          "ErrorMonitor",
          "L2MemoryManager",
          "L3GraphReasoner",
          "L4LLMInterface",
          "LLMConfig",
          "LLMProviderType",
          "MemoryConfig",
          "MemoryMode",
          "ScalableGraphBuilder"
        ],
        "function_names": [
          "get_llm_provider"
        ]
      },
      "insightspike.implementations.layers.cached_memory_manager": {
        "total": 5,
        "classes": 4,
        "functions": 1,
        "class_names": [
          "CachedMemoryManager",
          "DataStore",
          "EmbeddingManager",
          "Episode"
        ],
        "function_names": [
          "get_memory_monitor"
        ]
      },
      "insightspike.implementations.layers.layer1_conductor": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "L1Conductor"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer1_error_monitor": {
        "total": 8,
        "classes": 5,
        "functions": 3,
        "class_names": [
          "ErrorMonitor",
          "KnownUnknownAnalysis",
          "L1ErrorMonitorInterface",
          "LayerInput",
          "LayerOutput"
        ],
        "function_names": [
          "analyze_input",
          "get_config",
          "uncertainty"
        ]
      },
      "insightspike.implementations.layers.layer1_stream_processor": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "BoundaryDetector",
          "EmbeddingManager",
          "Episode",
          "L1Episode",
          "L1IntegratedMemory",
          "Layer1Config",
          "Layer1StreamProcessor"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer2_compatibility": {
        "total": 5,
        "classes": 4,
        "functions": 1,
        "class_names": [
          "CompatibleL2MemoryManager",
          "L2MemoryManager",
          "MemoryConfig",
          "MemoryMode"
        ],
        "function_names": [
          "create_compatible_memory"
        ]
      },
      "insightspike.implementations.layers.layer2_memory_manager": {
        "total": 13,
        "classes": 11,
        "functions": 2,
        "class_names": [
          "EmbeddingManager",
          "EnhancedL2MemoryManager",
          "Episode",
          "GraphCentricMemoryManager",
          "L2EnhancedScalableMemory",
          "L2MemoryManager",
          "Memory",
          "MemoryConfig",
          "MemoryMode",
          "ScalableGraphBuilder",
          "VectorIndexFactory"
        ],
        "function_names": [
          "create_memory_manager",
          "get_config"
        ]
      },
      "insightspike.implementations.layers.layer2_working_memory": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "DataStore",
          "EmbeddingManager",
          "L2WorkingMemoryManager",
          "WorkingMemoryConfig"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3": {
        "total": 10,
        "classes": 9,
        "functions": 1,
        "class_names": [
          "ConflictScore",
          "EdgeReevaluator",
          "GraphAnalyzer",
          "GraphBuilder",
          "L3GraphReasoner",
          "MessagePassing",
          "MessagePassingController",
          "MetricsController",
          "RewardCalculator"
        ],
        "function_names": [
          "build_simple_gnn"
        ]
      },
      "insightspike.implementations.layers.layer3.__main__": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "L3GraphReasoner"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3.analysis": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "GraphAnalyzer",
          "RewardCalculator"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3.analyzer_runner": {
        "total": 5,
        "classes": 3,
        "functions": 2,
        "class_names": [
          "GraphAnalyzer",
          "MessagePassingController",
          "RewardCalculator"
        ],
        "function_names": [
          "handle_query_focal",
          "run_analysis"
        ]
      },
      "insightspike.implementations.layers.layer3.conflict": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "ConflictScore",
          "LegacyConfigAdapter"
        ],
        "function_names": [
          "get_config"
        ]
      },
      "insightspike.implementations.layers.layer3.gnn": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "build_simple_gnn"
        ]
      },
      "insightspike.implementations.layers.layer3.graph_builder": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "GraphBuilder",
          "LegacyConfigAdapter"
        ],
        "function_names": [
          "get_config"
        ]
      },
      "insightspike.implementations.layers.layer3.lite_stub": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "L3GraphReasonerLiteStub"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3.message_passing": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "EdgeReevaluator",
          "MessagePassing"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3.message_passing_controller": {
        "total": 5,
        "classes": 4,
        "functions": 1,
        "class_names": [
          "EdgeReevaluator",
          "LegacyConfigAdapter",
          "MessagePassing",
          "MessagePassingController"
        ],
        "function_names": [
          "get_config"
        ]
      },
      "insightspike.implementations.layers.layer3.metrics_controller": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "MetricsController"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.layer3_graph_reasoner": {
        "total": 15,
        "classes": 9,
        "functions": 6,
        "class_names": [
          "ConflictScore",
          "Episode",
          "GraphBuilder",
          "L3GraphReasoner",
          "L3GraphReasonerInterface",
          "LayerInput",
          "LayerOutput",
          "LegacyConfigAdapter",
          "ScalableGraphBuilder"
        ],
        "function_names": [
          "advanced_delta_ged",
          "advanced_delta_ig",
          "cosine_similarity",
          "get_config",
          "simple_delta_ged",
          "simple_delta_ig"
        ]
      },
      "insightspike.implementations.layers.layer4_llm_interface": {
        "total": 10,
        "classes": 9,
        "functions": 1,
        "class_names": [
          "CleanLLMProvider",
          "L4LLMInterface",
          "LLMConfig",
          "LLMProvider",
          "LLMProviderRegistry",
          "LLMProviderType",
          "MockLLMProvider",
          "UnifiedLLMProvider",
          "VectorIntegrator"
        ],
        "function_names": [
          "get_llm_provider"
        ]
      },
      "insightspike.implementations.layers.layer4_prompt_builder": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "L4Interface",
          "L4PromptBuilder"
        ],
        "function_names": []
      },
      "insightspike.implementations.layers.scalable_graph_builder": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "GraphOperationMonitor",
          "LegacyConfigAdapter",
          "MonitoredOperation",
          "ScalableGraphBuilder",
          "VectorIndexFactory"
        ],
        "function_names": []
      },
      "insightspike.implementations.memory": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "DataStore",
          "EmbeddingManager",
          "InMemoryDataStore",
          "Memory"
        ],
        "function_names": []
      },
      "insightspike.implementations.memory.graph_memory_search": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "GraphMemorySearch"
        ],
        "function_names": []
      },
      "insightspike.implementations.memory.knowledge_graph_memory": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "KnowledgeGraphMemory"
        ],
        "function_names": []
      },
      "insightspike.implementations.memory.scalable_graph_manager": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "ScalableGraphManager",
          "VectorIndexFactory"
        ],
        "function_names": []
      },
      "insightspike.index": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "BackwardCompatibleWrapper",
          "IntegratedVectorGraphIndex",
          "MigrationHelper"
        ],
        "function_names": []
      },
      "insightspike.index.backward_compatible_wrapper": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "BackwardCompatibleWrapper",
          "IntegratedVectorGraphIndex"
        ],
        "function_names": []
      },
      "insightspike.index.integrated_vector_graph_index": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "IntegratedVectorGraphIndex"
        ],
        "function_names": []
      },
      "insightspike.index.migration_helper": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "IntegratedVectorGraphIndex",
          "MigrationHelper"
        ],
        "function_names": []
      },
      "insightspike.integrations": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "AssessmentResult",
          "EducationalSystemIntegration",
          "LearningPath",
          "Student"
        ],
        "function_names": []
      },
      "insightspike.integrations.educational": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "AssessmentResult",
          "EducationalSystemIntegration",
          "LearningPath",
          "Student"
        ],
        "function_names": []
      },
      "insightspike.interfaces": {
        "total": 10,
        "classes": 10,
        "functions": 0,
        "class_names": [
          "IAgent",
          "IDataStore",
          "IEmbedder",
          "IEpisodeStore",
          "IGraphAnalyzer",
          "IGraphBuilder",
          "ILLMProvider",
          "IMemoryAgent",
          "IMemoryManager",
          "IMemorySearch"
        ],
        "function_names": []
      },
      "insightspike.interfaces.agent": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "CycleResult",
          "IAgent",
          "IMemoryAgent"
        ],
        "function_names": []
      },
      "insightspike.interfaces.datastore": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "Episode",
          "IDataStore",
          "IEpisodeStore"
        ],
        "function_names": []
      },
      "insightspike.interfaces.embedder": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "IEmbedder"
        ],
        "function_names": []
      },
      "insightspike.interfaces.graph": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "Episode",
          "IGraphAnalyzer",
          "IGraphBuilder"
        ],
        "function_names": []
      },
      "insightspike.interfaces.llm": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "ILLMProvider"
        ],
        "function_names": []
      },
      "insightspike.interfaces.memory": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "Episode",
          "IMemoryManager",
          "IMemorySearch"
        ],
        "function_names": []
      },
      "insightspike.learning": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "PatternLogger",
          "ReasoningPattern",
          "StrategyOptimizer"
        ],
        "function_names": []
      },
      "insightspike.learning.pattern_logger": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "PatternLogger",
          "ReasoningPattern"
        ],
        "function_names": []
      },
      "insightspike.learning.strategy_optimizer": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "PatternLogger",
          "ReasoningPattern",
          "StrategyOptimizer"
        ],
        "function_names": []
      },
      "insightspike.metrics": {
        "total": 15,
        "classes": 2,
        "functions": 13,
        "class_names": [
          "GraphEditDistance",
          "InformationGain"
        ],
        "function_names": [
          "algo_delta_ged",
          "algo_delta_ig",
          "analyze_insight",
          "apply_preset_configuration",
          "compute_delta_ged",
          "compute_delta_ig",
          "compute_fusion_reward",
          "configure_default_weights",
          "delta_ged",
          "delta_ig",
          "get_algorithm_info",
          "get_algorithm_metadata",
          "get_preset_configurations"
        ]
      },
      "insightspike.metrics.graph_metrics": {
        "total": 5,
        "classes": 2,
        "functions": 3,
        "class_names": [
          "GraphEditDistance",
          "InformationGain"
        ],
        "function_names": [
          "delta_ged",
          "delta_ig",
          "get_global_ged_calculator"
        ]
      },
      "insightspike.metrics.improved_gedig_metrics": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "GEDIGMetrics",
          "ImprovedGEDIGCalculator"
        ],
        "function_names": [
          "calculate_gedig_metrics",
          "compute_gedig_legacy"
        ]
      },
      "insightspike.metrics.psz": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "PSZSummary",
          "PSZThresholds"
        ],
        "function_names": [
          "inside_psz",
          "summarize_accept_latency"
        ]
      },
      "insightspike.metrics.pyg_compatible_metrics": {
        "total": 3,
        "classes": 0,
        "functions": 3,
        "class_names": [],
        "function_names": [
          "delta_ged_pyg",
          "delta_ig_pyg",
          "pyg_to_networkx"
        ]
      },
      "insightspike.metrics.validation_helpers": {
        "total": 7,
        "classes": 5,
        "functions": 2,
        "class_names": [
          "GeDIGNavigator",
          "MazeNavigatorConfig",
          "MazeObservation",
          "SimpleMaze",
          "StabilityResult"
        ],
        "function_names": [
          "run_small_maze_stability",
          "run_spike_reproducibility"
        ]
      },
      "insightspike.monitoring": {
        "total": 6,
        "classes": 5,
        "functions": 1,
        "class_names": [
          "GraphOperationMetric",
          "GraphOperationMonitor",
          "IndexMonitoringDecorator",
          "IndexPerformanceMonitor",
          "MonitoredOperation"
        ],
        "function_names": [
          "create_default_monitor"
        ]
      },
      "insightspike.monitoring.graph_monitor": {
        "total": 4,
        "classes": 3,
        "functions": 1,
        "class_names": [
          "GraphOperationMetric",
          "GraphOperationMonitor",
          "MonitoredOperation"
        ],
        "function_names": [
          "create_default_monitor"
        ]
      },
      "insightspike.monitoring.index_monitor": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "IndexMonitoringDecorator",
          "IndexPerformanceMonitor"
        ],
        "function_names": []
      },
      "insightspike.monitoring.memory_monitor": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "MemoryMonitor",
          "MemorySnapshot"
        ],
        "function_names": [
          "check_memory_usage",
          "get_memory_monitor"
        ]
      },
      "insightspike.processing": {
        "total": 3,
        "classes": 0,
        "functions": 3,
        "class_names": [],
        "function_names": [
          "get_model",
          "load_corpus",
          "retrieve"
        ]
      },
      "insightspike.processing.embedder": {
        "total": 5,
        "classes": 2,
        "functions": 3,
        "class_names": [
          "EmbeddingManager",
          "FallbackEmbedder"
        ],
        "function_names": [
          "get_model",
          "normalize_batch_embeddings",
          "normalize_embedding_shape"
        ]
      },
      "insightspike.processing.loader": {
        "total": 4,
        "classes": 0,
        "functions": 4,
        "class_names": [],
        "function_names": [
          "clean_text",
          "get_config",
          "iter_text",
          "load_corpus"
        ]
      },
      "insightspike.processing.retrieval": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "retrieve"
        ]
      },
      "insightspike.processing.standardized_embedder": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "StandardizedEmbedder"
        ],
        "function_names": []
      },
      "insightspike.providers": {
        "total": 7,
        "classes": 7,
        "functions": 0,
        "class_names": [
          "AnthropicProvider",
          "DistilGPT2Provider",
          "LLMProviderRegistry",
          "LocalProvider",
          "MockProvider",
          "OpenAIProvider",
          "ProviderFactory"
        ],
        "function_names": []
      },
      "insightspike.providers.anthropic_provider": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "AnthropicProvider",
          "LLMConfig"
        ],
        "function_names": []
      },
      "insightspike.providers.distilgpt2_provider": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "DistilGPT2Provider",
          "LLMConfig"
        ],
        "function_names": []
      },
      "insightspike.providers.local": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "LLMConfig",
          "LocalProvider"
        ],
        "function_names": []
      },
      "insightspike.providers.mock_provider": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "LLMConfig",
          "MockProvider"
        ],
        "function_names": []
      },
      "insightspike.providers.openai_provider": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "LLMConfig",
          "OpenAIProvider"
        ],
        "function_names": []
      },
      "insightspike.providers.provider_factory": {
        "total": 8,
        "classes": 8,
        "functions": 0,
        "class_names": [
          "AnthropicProvider",
          "DistilGPT2Provider",
          "LLMConfig",
          "LocalProvider",
          "MockProvider",
          "OpenAIProvider",
          "ProviderFactory",
          "SimpleLocalProvider"
        ],
        "function_names": []
      },
      "insightspike.providers.simple_local": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "LLMConfig",
          "SimpleLocalProvider"
        ],
        "function_names": []
      },
      "insightspike.public": {
        "total": 6,
        "classes": 1,
        "functions": 5,
        "class_names": [
          "InsightAppWrapper"
        ],
        "function_names": [
          "create_agent",
          "create_datastore",
          "get_config_summary",
          "load_config",
          "quick_demo"
        ]
      },
      "insightspike.public.wrapper": {
        "total": 3,
        "classes": 2,
        "functions": 1,
        "class_names": [
          "InsightAppWrapper",
          "LLMConfig"
        ],
        "function_names": [
          "create_agent"
        ]
      },
      "insightspike.query_adaptation": {
        "total": 11,
        "classes": 6,
        "functions": 5,
        "class_names": [
          "AdaptiveTopKConfig",
          "Layer1AutoLearningSystem",
          "LearningSession",
          "UnknownConcept",
          "UnknownLearner",
          "WeakRelationship"
        ],
        "function_names": [
          "calculate_adaptive_topk",
          "estimate_chain_reaction_potential",
          "get_unknown_learner",
          "shutdown_unknown_learner",
          "test_adaptive_topk"
        ]
      },
      "insightspike.query_adaptation.adaptive_topk": {
        "total": 4,
        "classes": 1,
        "functions": 3,
        "class_names": [
          "AdaptiveTopKConfig"
        ],
        "function_names": [
          "calculate_adaptive_topk",
          "estimate_chain_reaction_potential",
          "test_adaptive_topk"
        ]
      },
      "insightspike.query_adaptation.auto_learning": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "Layer1AutoLearningSystem",
          "LearningSession",
          "UnknownConcept"
        ],
        "function_names": []
      },
      "insightspike.query_adaptation.unknown_learner": {
        "total": 4,
        "classes": 2,
        "functions": 2,
        "class_names": [
          "UnknownLearner",
          "WeakRelationship"
        ],
        "function_names": [
          "get_unknown_learner",
          "shutdown_unknown_learner"
        ]
      },
      "insightspike.quick_start": {
        "total": 4,
        "classes": 1,
        "functions": 3,
        "class_names": [
          "MainAgent"
        ],
        "function_names": [
          "create_agent",
          "load_config",
          "quick_demo"
        ]
      },
      "insightspike.spike_pipeline": {
        "total": 6,
        "classes": 6,
        "functions": 0,
        "class_names": [
          "SpikeDataCollector",
          "SpikeDecisionEngine",
          "SpikeDecisionMode",
          "SpikePipeline",
          "SpikePostProcessor",
          "SpikeStatsAnalyzer"
        ],
        "function_names": []
      },
      "insightspike.spike_pipeline.analyzer": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "SpikeDataCollection",
          "SpikeStats",
          "SpikeStatsAnalyzer"
        ],
        "function_names": []
      },
      "insightspike.spike_pipeline.collector": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "SpikeDataCollection",
          "SpikeDataCollector"
        ],
        "function_names": []
      },
      "insightspike.spike_pipeline.detector": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "SpikeDecision",
          "SpikeDecisionEngine",
          "SpikeDecisionMode",
          "SpikeStats"
        ],
        "function_names": []
      },
      "insightspike.spike_pipeline.pipeline": {
        "total": 13,
        "classes": 10,
        "functions": 3,
        "class_names": [
          "SpikeDataCollection",
          "SpikeDataCollector",
          "SpikeDecision",
          "SpikeDecisionEngine",
          "SpikeDecisionMode",
          "SpikePipeline",
          "SpikePostProcessor",
          "SpikeProcessingResult",
          "SpikeStats",
          "SpikeStatsAnalyzer"
        ],
        "function_names": [
          "create_adaptive_pipeline",
          "create_standard_pipeline",
          "create_threshold_pipeline"
        ]
      },
      "insightspike.spike_pipeline.processor": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "SpikeDataCollection",
          "SpikeDecision",
          "SpikePostProcessor",
          "SpikeProcessingResult",
          "SpikeStats"
        ],
        "function_names": []
      },
      "insightspike.training": {
        "total": 3,
        "classes": 0,
        "functions": 3,
        "class_names": [],
        "function_names": [
          "predict",
          "quantize",
          "train"
        ]
      },
      "insightspike.training.predict": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "predict"
        ]
      },
      "insightspike.training.quantizer": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "quantize"
        ]
      },
      "insightspike.training.train": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "train"
        ]
      },
      "insightspike.utils": {
        "total": 11,
        "classes": 3,
        "functions": 8,
        "class_names": [
          "DynamicImportanceTracker",
          "GraphImportanceCalculator",
          "InsightSpikeVisualizer"
        ],
        "function_names": [
          "clean_text",
          "delta_ged",
          "delta_ig",
          "get_available_models",
          "iter_text",
          "quick_comparison",
          "quick_performance_chart",
          "quick_progress_chart"
        ]
      },
      "insightspike.utils.config_access": {
        "total": 2,
        "classes": 0,
        "functions": 2,
        "class_names": [],
        "function_names": [
          "safe_attr",
          "safe_has"
        ]
      },
      "insightspike.utils.config_utils": {
        "total": 7,
        "classes": 0,
        "functions": 7,
        "class_names": [],
        "function_names": [
          "get_config_value",
          "is_dict_config",
          "is_object_config",
          "merge_configs",
          "normalize_to_dict",
          "safe_get",
          "set_config_value"
        ]
      },
      "insightspike.utils.embedding_utils": {
        "total": 3,
        "classes": 0,
        "functions": 3,
        "class_names": [],
        "function_names": [
          "normalize_batch_embeddings",
          "normalize_embedding_shape",
          "validate_embedding_dimension"
        ]
      },
      "insightspike.utils.path_utils": {
        "total": 1,
        "classes": 0,
        "functions": 1,
        "class_names": [],
        "function_names": [
          "resolve_project_relative"
        ]
      },
      "insightspike.utils.response_evaluator": {
        "total": 2,
        "classes": 2,
        "functions": 0,
        "class_names": [
          "ResponseEvaluator",
          "SimilarityConverter"
        ],
        "function_names": []
      },
      "insightspike.utils.similarity_converter": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "SimilarityConverter"
        ],
        "function_names": []
      },
      "insightspike.utils.text_utils": {
        "total": 3,
        "classes": 0,
        "functions": 3,
        "class_names": [],
        "function_names": [
          "clean_text",
          "iter_text",
          "jaccard_similarity"
        ]
      },
      "insightspike.vector_index": {
        "total": 4,
        "classes": 4,
        "functions": 0,
        "class_names": [
          "NumpyNearestNeighborIndex",
          "OptimizedNumpyIndex",
          "VectorIndexFactory",
          "VectorIndexInterface"
        ],
        "function_names": []
      },
      "insightspike.vector_index.factory": {
        "total": 5,
        "classes": 5,
        "functions": 0,
        "class_names": [
          "FaissIndexWrapper",
          "NumpyNearestNeighborIndex",
          "OptimizedNumpyIndex",
          "VectorIndexFactory",
          "VectorIndexInterface"
        ],
        "function_names": []
      },
      "insightspike.vector_index.interface": {
        "total": 1,
        "classes": 1,
        "functions": 0,
        "class_names": [
          "VectorIndexInterface"
        ],
        "function_names": []
      },
      "insightspike.vector_index.numpy_index": {
        "total": 3,
        "classes": 3,
        "functions": 0,
        "class_names": [
          "NumpyNearestNeighborIndex",
          "OptimizedNumpyIndex",
          "VectorIndexInterface"
        ],
        "function_names": []
      }
    },
    "top_modules": [
      [
        "insightspike.cli.spike",
        {
          "total": 28,
          "classes": 7,
          "functions": 21,
          "class_names": [
            "CLIState",
            "ConfigLoader",
            "ConfigPresets",
            "DependencyFactory",
            "InsightSpikeConfig",
            "InsightSpikeError",
            "MainAgent"
          ],
          "function_names": [
            "analyze_command",
            "bridge_command",
            "config",
            "demo",
            "discover_command",
            "embed",
            "experiment",
            "get_config",
            "get_logger",
            "insights",
            "insights_search",
            "interactive",
            "load_config",
            "main",
            "query",
            "run_cli",
            "show_stats",
            "sleep_gc_command",
            "stats",
            "version",
            "visualize_command"
          ]
        }
      ],
      [
        "insightspike.config.models",
        {
          "total": 24,
          "classes": 24,
          "functions": 0,
          "class_names": [
            "DataStoreConfig",
            "DefaultsAppliedMixin",
            "EmbeddingConfig",
            "GraphConfig",
            "HybridWeightsConfig",
            "InsightSpikeConfig",
            "LLMConfig",
            "LoggingConfig",
            "MazeConfig",
            "MazeExperimentConfig",
            "MazeNavigatorConfig",
            "MemoryConfig",
            "MetricsConfig",
            "MonitoringConfig",
            "MultihopConfig",
            "OutputConfig",
            "PathsConfig",
            "PerformanceConfig",
            "ProcessingConfig",
            "ReasoningConfig",
            "SpectralEvaluationConfig",
            "StructuralSimilarityConfig",
            "VectorSearchConfig",
            "WakeSleepConfig"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.core.base",
        {
          "total": 23,
          "classes": 23,
          "functions": 0,
          "class_names": [
            "ActionSpace",
            "AgentInterface",
            "EnvironmentInterface",
            "EnvironmentState",
            "GenericAgentInterface",
            "InsightDetectorInterface",
            "InsightMoment",
            "L1ErrorMonitorInterface",
            "L2MemoryInterface",
            "L3GraphReasonerInterface",
            "L4LLMInterface",
            "LayerInput",
            "LayerInterface",
            "LayerOutput",
            "MazeEnvironmentAdapter",
            "MazeInsightDetector",
            "MazeRewardNormalizer",
            "MazeStateEncoder",
            "MemoryManagerInterface",
            "ReasonerInterface",
            "RewardNormalizer",
            "StateEncoder",
            "TaskType"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike",
        {
          "total": 22,
          "classes": 13,
          "functions": 9,
          "class_names": [
            "About",
            "AgentConfigBuilder",
            "CycleResult",
            "EnvironmentInterface",
            "ErrorMonitor",
            "GenericInsightSpikeAgent",
            "InsightMoment",
            "InsightSpikeAgentFactory",
            "L2MemoryManager",
            "L3GraphReasoner",
            "MainAgent",
            "StandaloneL3GraphReasoner",
            "TaskType"
          ],
          "function_names": [
            "about",
            "analyze_documents_simple",
            "create_agent",
            "create_configured_maze_agent",
            "create_maze_agent",
            "create_standalone_reasoner",
            "get_config",
            "get_llm_provider",
            "quick_demo"
          ]
        }
      ],
      [
        "insightspike.algorithms",
        {
          "total": 22,
          "classes": 9,
          "functions": 13,
          "class_names": [
            "ContentStructureSeparation",
            "EntropyCalculator",
            "EntropyMethod",
            "EntropyResult",
            "GEDResult",
            "GraphEditDistance",
            "IGResult",
            "InformationGain",
            "OptimizationLevel"
          ],
          "function_names": [
            "clustering_coefficient_entropy",
            "compute_delta_ged",
            "compute_delta_ig",
            "compute_graph_edit_distance",
            "compute_information_gain",
            "compute_shannon_entropy",
            "create_default_ged_calculator",
            "create_default_ig_calculator",
            "degree_distribution_entropy",
            "get_algorithm_info",
            "path_length_entropy",
            "structural_entropy",
            "von_neumann_entropy"
          ]
        }
      ],
      [
        "insightspike.core.exceptions",
        {
          "total": 22,
          "classes": 22,
          "functions": 0,
          "class_names": [
            "AgentError",
            "AgentInitializationError",
            "AgentProcessingError",
            "ConfigNotFoundError",
            "ConfigurationError",
            "DataStoreError",
            "DataStoreLoadError",
            "DataStoreNotFoundError",
            "DataStorePermissionError",
            "DataStoreSaveError",
            "GraphAnalysisError",
            "GraphBuildError",
            "GraphError",
            "InsightSpikeException",
            "InvalidConfigError",
            "LLMConnectionError",
            "LLMError",
            "LLMGenerationError",
            "LLMTokenLimitError",
            "MemoryCapacityError",
            "MemoryError",
            "MemorySearchError"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.implementations.agents.main_agent",
        {
          "total": 20,
          "classes": 15,
          "functions": 5,
          "class_names": [
            "AgentConfig",
            "AgentMode",
            "ConfigurableAgent",
            "CycleResult",
            "DataStore",
            "Episode",
            "ErrorMonitor",
            "GeDIGABLogger",
            "GeDIGFallbackTracker",
            "GraphMemorySearch",
            "InsightSpikeConfig",
            "L3GraphReasoner",
            "MainAgent",
            "Memory",
            "TwoThresholdCandidateSelector"
          ],
          "function_names": [
            "cycle",
            "get_insight_registry",
            "get_llm_provider",
            "safe_attr",
            "safe_has"
          ]
        }
      ],
      [
        "insightspike.di.providers",
        {
          "total": 16,
          "classes": 16,
          "functions": 0,
          "class_names": [
            "DataStoreFactory",
            "DataStoreProvider",
            "EmbedderProvider",
            "EmbeddingManager",
            "GraphBuilderProvider",
            "IDataStore",
            "IEmbedder",
            "IGraphBuilder",
            "ILLMProvider",
            "IMemoryManager",
            "InsightSpikeConfig",
            "L2MemoryManager",
            "LLMProviderFactory",
            "LLMProviderRegistry",
            "MemoryManagerProvider",
            "PyGGraphBuilder"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.implementations.agents.slim_main_agent",
        {
          "total": 15,
          "classes": 11,
          "functions": 4,
          "class_names": [
            "CycleResult",
            "DataStore",
            "Episode",
            "ErrorMonitor",
            "FallbackReason",
            "GraphMemorySearch",
            "L3GraphReasoner",
            "Memory",
            "SlimMainAgent",
            "SpikeDecisionMode",
            "SpikePipeline"
          ],
          "function_names": [
            "create_slim_agent",
            "execute_fallback",
            "get_fallback_registry",
            "get_insight_registry"
          ]
        }
      ],
      [
        "insightspike.implementations.layers.layer3_graph_reasoner",
        {
          "total": 15,
          "classes": 9,
          "functions": 6,
          "class_names": [
            "ConflictScore",
            "Episode",
            "GraphBuilder",
            "L3GraphReasoner",
            "L3GraphReasonerInterface",
            "LayerInput",
            "LayerOutput",
            "LegacyConfigAdapter",
            "ScalableGraphBuilder"
          ],
          "function_names": [
            "advanced_delta_ged",
            "advanced_delta_ig",
            "cosine_similarity",
            "get_config",
            "simple_delta_ged",
            "simple_delta_ig"
          ]
        }
      ],
      [
        "insightspike.metrics",
        {
          "total": 15,
          "classes": 2,
          "functions": 13,
          "class_names": [
            "GraphEditDistance",
            "InformationGain"
          ],
          "function_names": [
            "algo_delta_ged",
            "algo_delta_ig",
            "analyze_insight",
            "apply_preset_configuration",
            "compute_delta_ged",
            "compute_delta_ig",
            "compute_fusion_reward",
            "configure_default_weights",
            "delta_ged",
            "delta_ig",
            "get_algorithm_info",
            "get_algorithm_metadata",
            "get_preset_configurations"
          ]
        }
      ],
      [
        "insightspike.algorithms.gedig_core",
        {
          "total": 13,
          "classes": 9,
          "functions": 4,
          "class_names": [
            "GeDIGCore",
            "GeDIGLogger",
            "GeDIGMonitor",
            "GeDIGPresets",
            "GeDIGResult",
            "HopResult",
            "LinksetMetrics",
            "ProcessingMode",
            "SpikeDetectionMode"
          ],
          "function_names": [
            "calculate_gedig",
            "delta_ged",
            "delta_ig",
            "detect_insight_spike"
          ]
        }
      ],
      [
        "insightspike.implementations.agents.generic_agent",
        {
          "total": 13,
          "classes": 13,
          "functions": 0,
          "class_names": [
            "EnvironmentInterface",
            "EnvironmentState",
            "GenericAgentInterface",
            "GenericInsightSpikeAgent",
            "GenericMemoryManager",
            "GenericReasoner",
            "InsightDetectorInterface",
            "InsightMoment",
            "MemoryManagerInterface",
            "ReasonerInterface",
            "RewardNormalizer",
            "StateEncoder",
            "TaskType"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.implementations.layers.layer2_memory_manager",
        {
          "total": 13,
          "classes": 11,
          "functions": 2,
          "class_names": [
            "EmbeddingManager",
            "EnhancedL2MemoryManager",
            "Episode",
            "GraphCentricMemoryManager",
            "L2EnhancedScalableMemory",
            "L2MemoryManager",
            "Memory",
            "MemoryConfig",
            "MemoryMode",
            "ScalableGraphBuilder",
            "VectorIndexFactory"
          ],
          "function_names": [
            "create_memory_manager",
            "get_config"
          ]
        }
      ],
      [
        "insightspike.spike_pipeline.pipeline",
        {
          "total": 13,
          "classes": 10,
          "functions": 3,
          "class_names": [
            "SpikeDataCollection",
            "SpikeDataCollector",
            "SpikeDecision",
            "SpikeDecisionEngine",
            "SpikeDecisionMode",
            "SpikePipeline",
            "SpikePostProcessor",
            "SpikeProcessingResult",
            "SpikeStats",
            "SpikeStatsAnalyzer"
          ],
          "function_names": [
            "create_adaptive_pipeline",
            "create_standard_pipeline",
            "create_threshold_pipeline"
          ]
        }
      ],
      [
        "insightspike.algorithms.metrics_selector",
        {
          "total": 12,
          "classes": 3,
          "functions": 9,
          "class_names": [
            "GraphEditDistance",
            "InformationGain",
            "MetricsSelector"
          ],
          "function_names": [
            "advanced_delta_ged",
            "advanced_delta_ig",
            "delta_ged",
            "delta_ged_pyg",
            "delta_ig",
            "delta_ig_pyg",
            "get_metrics_selector",
            "simple_delta_ged",
            "simple_delta_ig"
          ]
        }
      ],
      [
        "insightspike.core.base.maze_implementation",
        {
          "total": 12,
          "classes": 12,
          "functions": 0,
          "class_names": [
            "ActionSpace",
            "EnvironmentInterface",
            "EnvironmentState",
            "InsightDetectorInterface",
            "InsightMoment",
            "MazeEnvironmentAdapter",
            "MazeInsightDetector",
            "MazeRewardNormalizer",
            "MazeStateEncoder",
            "RewardNormalizer",
            "StateEncoder",
            "TaskType"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.features.query_transformation",
        {
          "total": 12,
          "classes": 12,
          "functions": 0,
          "class_names": [
            "AdaptiveExplorer",
            "EnhancedQueryTransformer",
            "EvolutionPattern",
            "EvolutionTracker",
            "MultiHopGNN",
            "PatternDatabase",
            "QueryBranch",
            "QueryState",
            "QueryTransformationHistory",
            "QueryTransformer",
            "QueryTypeClassifier",
            "TrajectoryAnalyzer"
          ],
          "function_names": []
        }
      ],
      [
        "insightspike.cli.commands.graph",
        {
          "total": 11,
          "classes": 4,
          "functions": 7,
          "class_names": [
            "DataStoreConfig",
            "DataStoreFactory",
            "GraphAnalyzer",
            "SimpleRAGGraph"
          ],
          "function_names": [
            "analyze_command",
            "graph_command",
            "load_config",
            "resolve_project_relative",
            "sleep_gc_command",
            "visualize_command",
            "visualize_graph_metrics"
          ]
        }
      ],
      [
        "insightspike.core.base.generic_interfaces",
        {
          "total": 11,
          "classes": 11,
          "functions": 0,
          "class_names": [
            "ActionSpace",
            "EnvironmentInterface",
            "EnvironmentState",
            "GenericAgentInterface",
            "InsightDetectorInterface",
            "InsightMoment",
            "MemoryManagerInterface",
            "ReasonerInterface",
            "RewardNormalizer",
            "StateEncoder",
            "TaskType"
          ],
          "function_names": []
        }
      ]
    ]
  },
  "naming_patterns": {
    "prefix_distribution": {
      "get_": 45,
      "create_": 27,
      "compute_": 17,
      "calculate_": 14,
      "build_": 3,
      "is_": 3,
      "validate_": 2,
      "set_": 1
    },
    "suffix_distribution": {
      "_config": 30,
      "_error": 1,
      "_manager": 1
    },
    "name_length_stats": {
      "mean": 15.830769230769231,
      "std": 5.361616039740751,
      "min": 3,
      "max": 38,
      "median": 16.0
    }
  },
  "documentation": {
    "documented_count": 1064,
    "undocumented_count": 41,
    "coverage_percentage": 96.289592760181,
    "doc_length_stats": {
      "mean": 53.94642857142857,
      "max": 1242
    }
  },
  "signature_complexity": {
    "param_count_stats": {
      "mean": 4.885067873303168,
      "std": 8.071199208890917,
      "max": 74
    },
    "simple_signatures": 700,
    "complex_signatures": 405,
    "complexity_ratio": 0.3665158371040724
  },
  "type_hierarchy": {
    "pattern_distribution": {
      "Config": 108,
      "Provider": 47,
      "Agent": 41,
      "Result": 41,
      "Error": 38,
      "Interface": 37,
      "Manager": 32,
      "Builder": 19,
      "Processor": 7
    },
    "mixin_count": 2,
    "abstract_count": 5,
    "factory_count": 21
  },
  "scripts": {
    "count": 32,
    "scripts": [
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/aggregate_baseline_results.py",
        "name": "aggregate_baseline_results.py",
        "size": 1339,
        "docstring": "Aggregate geDIG and baseline metrics for quick comparison.  from __future__ import annotations  import argparse import json from dataclasses import dataclass from pathlib import Path from typing import Dict   @dataclass class Metrics: per_mean: float acceptance_rate: float fmr: float latency_p50: float   def load_metrics(path: Path, key: str) -> Metrics: data = json.loads(path.read_text()) result = data[\"results\"][key] return Metrics( per_mean=float(result[\"per_mean\"]), acceptance_rate=float(result[\"acceptance_rate\"]), fmr=float(result[\"fmr\"]), latency_p50=float(result[\"latency_p50\"]), )   def main() -> None: ap = argparse.ArgumentParser() ap.add_argument(\"--gedig\", type=Path, required=True, help=\"geDIG results JSON\") ap.add_argument(\"--baseline\", type=Path, required=True, help=\"baseline results JSON\") ap.add_argument(\"--baseline-key\", type=str, default=\"graphrag_baseline\") args = ap.parse_args()  gedig = load_metrics(args.gedig, \"gedig_ag_dg\") baseline = load_metrics(args.baseline, args.baseline_key)  table: Dict[str, Dict[str, float]] = { \"geDIG\": gedig.__dict__, \"baseline\": baseline.__dict__, } print(json.dumps(table, indent=2))   if __name__ == \"__main__\": main()"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/aggregate_maze_batch.py",
        "name": "aggregate_maze_batch.py",
        "size": 3634,
        "docstring": "Aggregate maze batch results into a single JSON for paper tables.  - Scans a directory for per-seed summary/step logs e.g., paper25_25x25_s500_seed*_summary.json and *_steps.json - Computes mean of key metrics across seeds - Optionally computes AG/DG rates from step logs (fraction of steps with ag_fire/dg_fire) - Emits a compact JSON suitable for docs/paper/data/*.json  Usage: python scripts/aggregate_maze_batch.py \\ --dir experiments/maze-query-hub-prototype/results/batch_25x25 \\ --prefix paper25_25x25_s500_eval_seed \\ --out docs/paper/data/maze_25x25_eval_s500.json  python scripts/aggregate_maze_batch.py \\ --dir experiments/maze-query-hub-prototype/results/batch_25x25 \\ --prefix paper25_25x25_s500_seed \\ --out docs/paper/data/maze_25x25_l3_s500.json"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/aggregate_maze_lambda_sweep.py",
        "name": "aggregate_maze_lambda_sweep.py",
        "size": 5545,
        "docstring": "Aggregate maze query-hub summaries by lambda and optionally plot.  Reads per-run summary JSONs produced by `experiments/maze-query-hub-prototype/run_experiment_query.py` and groups them by `lambda_weight` (taken from summary \u2192 config fallback).  Example: python scripts/aggregate_maze_lambda_sweep.py \\ --dir results/maze-lambda-sweep \\ --glob \"*_summary.json\" \\ --out results/maze-lambda-sweep/maze_lambda_agg.json \\ --plot results/maze-lambda-sweep/maze_lambda_plot.png"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/aggregate_rag_lambda_sweep.py",
        "name": "aggregate_rag_lambda_sweep.py",
        "size": 5167,
        "docstring": "Aggregate exp2to4_lite (RAG lite) results by lambda and optionally plot.  Reads JSON outputs from experiments/exp2to4_lite/src/run_suite.py and groups the chosen baseline metrics by `lambda_weight`.  Example: python scripts/aggregate_rag_lambda_sweep.py \\ --dir experiments/exp2to4_lite/results \\ --glob \"exp23_paper_lambda*_*.json\" \\ --baseline gedig_ag_dg \\ --out results/rag-lambda/agg.json \\ --plot results/rag-lambda/plot.png"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/analyze_ab_k_stats.py",
        "name": "analyze_ab_k_stats.py",
        "size": 2005,
        "docstring": "Analyze A/B geDIG k-estimates aggregated over one or more CSV logs.  Expected CSV header fields (minimal subset used by tests): - query_id, pure_gedig, full_gedig, pure_ged, full_ged, pure_ig, full_ig, k_estimate, k_missing_reason, window_corr_at_record, timestamp  Public function: compute_k_stats(paths: list[str]) -> dict Returns: { 'total_rows': int, 'rows_with_k': int, 'missing_rate': float, 'k_min': float|None, 'k_max': float|None, 'window_corr_last': float|None, }"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/analyze_step_logs.py",
        "name": "analyze_step_logs.py",
        "size": 2070,
        "docstring": ""
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/audit_large_files.py",
        "name": "audit_large_files.py",
        "size": 1614,
        "docstring": "List large tracked files to help decide Releases/LFS moves.  Usage: python scripts/audit_large_files.py --min-mb 10 --top 200 > docs/asset_audit/LARGE_FILES.md"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/build_plan_index.py",
        "name": "build_plan_index.py",
        "size": 866,
        "docstring": "Legacy docs index builder stub.  The original project used this script to maintain an index under docs/development/. That directory no longer exists in this fork, but GitHub Actions still calls this script via docs_lint.yml. To keep CI green without reintroducing the old system, we provide a no-op stub that exits successfully."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/check_g0_snapshot.py",
        "name": "check_g0_snapshot.py",
        "size": 4257,
        "docstring": "Quick sanity checker for maze run outputs.  Loads a run JSON (either raw output from run_experiment.py or the derived visualization run_data.json) and verifies that:  * g0_history aligns with g0_components[*]['g0'] * gmin_history aligns with g0_components[*]['gmin'] when present * IG-related fields are populated when ig_value != 0  Intended for lightweight regression checks after navigator/geDIG changes."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/convert_for_baselines.py",
        "name": "convert_for_baselines.py",
        "size": 3690,
        "docstring": "Convert RAG JSONL datasets into baseline-friendly formats.  Currently supports: - GraphRAG: documents.tsv + questions.jsonl  Usage: python scripts/convert_for_baselines.py \\ --input data/sample_queries.jsonl \\ --output-dir experiments/rag-baselines-data/graph_rag"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/decode_results.py",
        "name": "decode_results.py",
        "size": 1291,
        "docstring": ""
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/doc_issue_scheduler.py",
        "name": "doc_issue_scheduler.py",
        "size": 711,
        "docstring": "Legacy documentation scheduler stub.  The original project generated a decision schedule from docs/development/*.md. That directory is not part of this repository, but the docs_lint workflow still invokes this script. We emit a tiny placeholder JSONL entry so downstream steps have predictable output."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/doc_meta_update.py",
        "name": "doc_meta_update.py",
        "size": 551,
        "docstring": "Legacy doc meta update stub.  The original project used this script to update front matter and indices for docs/development/*.md. The current repository does not ship those docs, but GitHub Actions still invokes this script.  To avoid CI failures while keeping behaviour benign, this stub simply prints a short message and exits with status 0."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/maze_phase1_replay.py",
        "name": "maze_phase1_replay.py",
        "size": 3844,
        "docstring": "Utility runner for Maze Phase-1 replays with optional feature profiles.  from __future__ import annotations  import argparse import subprocess import sys from pathlib import Path from typing import List   def build_command( *, python_bin: str, runner: Path, size: int, seeds: int, seed_offset: int, max_steps: int, feature_profile: str, output_path: Path, summary_path: Path, log_pattern: Path, compare_baseline: bool, extra_args: List[str], ) -> List[str]: cmd: List[str] = [ python_bin, str(runner), \"--size\", str(size), \"--seeds\", str(seeds), \"--seed-offset\", str(seed_offset), \"--max-steps\", str(max_steps), \"--feature-profile\", feature_profile, \"--summary\", str(summary_path), \"--output\", str(output_path), \"--log-steps\", str(log_pattern), ] if compare_baseline: cmd.append(\"--compare-baseline\") if extra_args: cmd.extend(extra_args) return cmd   def main() -> None: parser = argparse.ArgumentParser(description=\"Maze Phase-1 replay orchestrator\") parser.add_argument(\"--size\", type=int, default=15, help=\"maze size (odd)\") parser.add_argument(\"--seeds\", type=int, default=5, help=\"number of seeds per profile\") parser.add_argument(\"--seed-offset\", type=int, default=0, help=\"seed offset\") parser.add_argument(\"--max-steps\", type=int, default=200, help=\"maximum steps per episode\") parser.add_argument( \"--feature-profiles\", nargs=\"+\", default=[\"default\"], choices=[\"default\", \"option_a\", \"option_b\"], help=\"feature profiles to evaluate\", ) parser.add_argument( \"--output-dir\", type=Path, default=Path(\"experiments/maze-online-phase1-querylog/results/replays\"), help=\"base directory for outputs\", ) parser.add_argument(\"--compare-baseline\", action=\"store_true\", help=\"run baseline heuristic\") parser.add_argument( \"--extra-args\", nargs=argparse.REMAINDER, help=\"additional arguments forwarded to run_experiment.py\", ) parser.add_argument(\"--dry-run\", action=\"store_true\", help=\"print commands without executing\") args = parser.parse_args()  runner = Path(__file__).resolve().parents[1] / \"experiments\" / \"maze-online-phase1-querylog\" / \"run_experiment.py\" if not runner.exists(): raise FileNotFoundError(f\"run_experiment.py not found at {runner}\")  output_dir = args.output_dir.resolve() log_dir = output_dir / \"step_logs\" output_dir.mkdir(parents=True, exist_ok=True) log_dir.mkdir(parents=True, exist_ok=True)  python_bin = sys.executable extra_args = args.extra_args or []  for profile in args.feature_profiles: summary_path = output_dir / f\"{profile}_summary.csv\" output_path = output_dir / f\"{profile}_raw.json\" log_pattern = log_dir / f\"{profile}_seed{{seed}}.csv\" cmd = build_command( python_bin=python_bin, runner=runner, size=args.size, seeds=args.seeds, seed_offset=args.seed_offset, max_steps=args.max_steps, feature_profile=profile, output_path=output_path, summary_path=summary_path, log_pattern=log_pattern, compare_baseline=args.compare_baseline, extra_args=extra_args, ) if args.dry_run: print(\"DRY:\", \" \".join(cmd)) continue result = subprocess.run(cmd, check=False) if result.returncode != 0: raise RuntimeError(f\"Command failed for profile {profile} (exit {result.returncode})\")   if __name__ == \"__main__\": main()"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/migrate_datastore.py",
        "name": "migrate_datastore.py",
        "size": 6836,
        "docstring": "Datastore Path Migration Utility ================================  Safely copy or move data from the legacy base path (typically \"data/\") to the new datastore root (typically \"./data/insight_store\").  Usage examples:  # Auto-detect from current config (source=paths.data_dir, dest=datastore.root_path) PYTHONPATH=src python scripts/migrate_datastore.py --mode copy --dry-run  # Execute copy PYTHONPATH=src python scripts/migrate_datastore.py --mode copy  # Move instead of copy PYTHONPATH=src python scripts/migrate_datastore.py --mode move  # Override paths explicitly PYTHONPATH=src python scripts/migrate_datastore.py \\ --source ./data \\ --dest ./data/insight_store \\ --mode copy"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/plot_step_log_hist.py",
        "name": "plot_step_log_hist.py",
        "size": 4423,
        "docstring": ""
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/run_baseline_compare.py",
        "name": "run_baseline_compare.py",
        "size": 4763,
        "docstring": "Generate a tiny baseline comparison (Static RAG vs geDIG) from existing lite experiment results under experiments/exp2to4_lite/results, and optionally update docs/phase1.md with a small Markdown table.  Usage: python scripts/run_baseline_compare.py \\ --results experiments/exp2to4_lite/results/exp23_lite_*.json \\ --out results/baseline/baseline_compare.json \\ --update-docs  Notes: - Prefers the most recent exp23_lite_*.json if --results is not provided. - Does not run any experiments; reads already-produced JSON to remain cloud-safe. - Updates docs/phase1.md between markers: <!-- BASELINE:BEGIN --> ... <!-- BASELINE:END -->"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/run_fixed_mazes.py",
        "name": "run_fixed_mazes.py",
        "size": 6301,
        "docstring": ""
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/run_large_scale.py",
        "name": "run_large_scale.py",
        "size": 1886,
        "docstring": ""
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/run_maze_batch_and_update.py",
        "name": "run_maze_batch_and_update.py",
        "size": 7651,
        "docstring": "Run 25x25 / s500 maze batch (eval or L3) across multiple seeds, then aggregate and optionally update the TeX table for the paper.  Examples: # Run 60 seeds for Eval path (use_main_l3 = false), aggregate, update TeX python scripts/run_maze_batch_and_update.py --mode eval --seeds 60 --workers 4 --update-tex  # Run 60 seeds for L3 path (use_main_l3 = true) and aggregate only python scripts/run_maze_batch_and_update.py --mode l3 --seeds 60 --workers 4  Notes: - This script calls the experiment driver: experiments/maze-query-hub-prototype/run_experiment_query.py - Outputs are written under batch_25x25 with the prefix: Eval: paper25_25x25_s500_eval_seed{seed} L3:   paper25_25x25_s500_seed{seed} - Aggregation uses scripts/aggregate_maze_batch.py and writes to: docs/paper/data/maze_25x25_eval_s500.json (Eval) docs/paper/data/maze_25x25_l3_s500.json   (L3) - When --update-tex is set, the script updates the Eval or L3 column in docs/paper/figures/maze_25x25_s500_table.tex accordingly."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/selftest_ab_logger.py",
        "name": "selftest_ab_logger.py",
        "size": 1514,
        "docstring": "Lightweight self-test for GeDIG A/B logger.  Cloud-safe: no network, no heavy deps, writes only under results/. Used by Makefile target `selftest-ab` and optional in scripts/codex_smoke.sh."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/structural_cost_migration.py",
        "name": "structural_cost_migration.py",
        "size": 4767,
        "docstring": "Convert legacy geDIG logs that still use structural_improvement.  Usage: python scripts/structural_cost_migration.py --input old.json --output new.json python scripts/structural_cost_migration.py --input step_log.csv --overwrite  Supports JSON (nested dict/list) and CSV (step logs). For JSON the script copies structural_improvement \u2192 structural_cost (if missing) and negates the value so the result matches the new positive-cost convention. The legacy key is preserved unless --drop-legacy is specified. For CSV the script writes a new column and optionally drops the legacy column."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/scripts/visualize_maze_gif.py",
        "name": "visualize_maze_gif.py",
        "size": 8968,
        "docstring": "Generate an animated GIF of a maze run from a step log.  Inputs - steps: JSON list where each element is a step record containing at least { \"position\": [row, col], ... } - layout (optional): JSON with key \"layout\" holding a 2D int grid (0=open, 1=wall, 2=start, 3=goal) - If layout is omitted, grid size is inferred from the first step using ring_Rr/ring_Rc or from the maximum position seen.  Example python scripts/visualize_maze_gif.py \\ --steps experiments/maze-query-hub-prototype/results/batch_25x25/paper25_25x25_s500_seed0_steps.json \\ --layout docs/paper/data/maze_25x25.json \\ --out docs/images/maze_demo.gif --cell 16 --fps 10  Notes - Requires Pillow (PIL). This script is for documentation/social assets and is not used in CI."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/config_examples.py",
        "name": "config_examples.py",
        "size": 8901,
        "docstring": "Configuration Examples for InsightSpike ======================================  Shows how to use the new Pydantic-based configuration system."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/hello_gating.py",
        "name": "hello_gating.py",
        "size": 1323,
        "docstring": "Minimal example: compute g0/gmin and AG/DG. from __future__ import annotations  import networkx as nx  from insightspike.algorithms.gedig_core import GeDIGCore from insightspike.algorithms.linkset_adapter import build_linkset_info from insightspike.algorithms.gating import decide_gates   def tiny_graphs(): g_before = nx.Graph() g_before.add_edges_from([(\"A\", \"B\"), (\"B\", \"C\")]) g_after = nx.Graph() g_after.add_edges_from([(\"A\", \"B\"), (\"B\", \"C\"), (\"A\", \"C\")]) return g_before, g_after   def main() -> None: g1, g2 = tiny_graphs() core = GeDIGCore(enable_multihop=True, max_hops=2, lambda_weight=1.0, ig_mode=\"norm\") ls = build_linkset_info( s_link=[{\"index\": 1, \"similarity\": 1.0}], candidate_pool=[], decision={\"index\": 1, \"similarity\": 1.0}, query_vector=[1.0], base_mode=\"link\", ) res = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)  if res.hop_results and 0 in res.hop_results: g0 = float(res.hop_results[0].gedig) else: g0 = float(res.gedig_value) gmin = float(res.gedig_value)  gate = decide_gates(g0=g0, gmin=gmin, theta_ag=0.5, theta_dg=0.0) print(f\"g0={gate.g0:.3f}, gmin={gate.gmin:.3f}, AG={gate.ag}, DG={gate.dg}, b(t)={gate.b_value:.3f}\")   if __name__ == \"__main__\": main()"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/hello_insight.py",
        "name": "hello_insight.py",
        "size": 1412,
        "docstring": "Minimal geDIG gauge demo.  Builds two tiny graphs and computes F = \u0394EPC_norm \u2212 \u03bb\u00b7\u0394IG. Prints F, \u0394EPC_norm, \u0394IG, and spike flag."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/playground.py",
        "name": "playground.py",
        "size": 4521,
        "docstring": ")  # --- Sidebar Controls --- st.sidebar.header(\"\ud83c\udf9b\ufe0f Parameters\")  lambda_val = st.sidebar.slider( \"Lambda (\u03bb) - The 'Skepticism' Factor\", min_value=0.0, max_value=2.0, value=1.0, step=0.1, help=\"High Lambda = Conservative (Hard to impress).\\nLow Lambda = Open-minded (Easily accepts updates).\" )  st.sidebar.markdown(\"---\") st.sidebar.header(\"Scenario\") scenario = st.sidebar.radio( \"Choose a topology:\", [\"The Shortcut (Classic)\", \"The Noise (Bad Edge)\", \"The Bridge (Community Merge)\"] )  # --- Graph Logic ---  def get_graphs(scenario_name): G_before = nx.Graph() G_after = nx.Graph()  if scenario_name == \"The Shortcut (Classic)\": # A long chain vs a direct shortcut # Before: A-B-C-D-E edges_b = [(\"A\", \"B\"), (\"B\", \"C\"), (\"C\", \"D\"), (\"D\", \"E\")] G_before.add_edges_from(edges_b)  # After: Add A-E directly G_after.add_edges_from(edges_b) G_after.add_edge(\"A\", \"E\")  desc = \"We found a direct path (A-E) that skips 3 hops.\" delta_epc = 0.2  # Mock cost delta_ig = 0.8   # High value (shortcut)  elif scenario_name == \"The Noise (Bad Edge)\": # A triangle edges_b = [(\"A\", \"B\"), (\"B\", \"C\"), (\"C\", \"A\")] G_before.add_edges_from(edges_b)  # Add a random leaf G_after.add_edges_from(edges_b) G_after.add_edge(\"C\", \"D\")  desc = \"Adding a leaf node (D). Does not improve overall connectivity much.\" delta_epc = 0.1  # Low cost delta_ig = 0.05  # Very low value  else: # The Bridge # Two clusters edges_b = [(\"A\",\"B\"), (\"B\",\"C\"), (\"C\",\"A\"), (\"D\",\"E\"), (\"E\",\"F\"), (\"F\",\"D\")] G_before.add_edges_from(edges_b)  # Connect them G_after.add_edges_from(edges_b) G_after.add_edge(\"C\", \"D\")  desc = \"Connecting two separate islands of knowledge.\" delta_epc = 0.3  # Moderate cost delta_ig = 0.9   # Huge value (global connectivity)  return G_before, G_after, desc, delta_epc, delta_ig  G_b, G_a, description, d_epc, d_ig = get_graphs(scenario)  # --- Calculation --- F = d_epc - (lambda_val * d_ig) is_spike = F < 0  # --- Visualization ---  col1, col2, col3 = st.columns([1, 2, 1])  with col1: st.subheader(\"Current Graph\") fig1, ax1 = plt.subplots(figsize=(4, 4)) pos1 = nx.spring_layout(G_b, seed=42) nx.draw(G_b, pos1, with_labels=True, node_color='lightblue', ax=ax1) st.pyplot(fig1)  with col2: st.subheader(\"The Decision Gauge (F)\")  # Gauge Metrics m1, m2, m3 = st.columns(3) m1.metric(\"Cost (\u0394EPC)\", f\"{d_epc:.2f}\") m2.metric(\"Gain (\u0394IG)\", f\"{d_ig:.2f}\") m3.metric(\"Result (F)\", f\"{F:.2f}\", delta=\"-Spike!\" if is_spike else \"Reject\", delta_color=\"inverse\")  st.info(description)  # Progress bar visualization of the equation st.markdown(\"### Balance\") cost_pct = min(1.0, d_epc) gain_scaled = min(1.0, lambda_val * d_ig)  st.text(f\"Structure Cost: {'\u2588' * int(cost_pct * 20)} ({d_epc:.2f})\") st.text(f\"Info Gain     : {'\u2592' * int(gain_scaled * 20)} ({lambda_val * d_ig:.2f})\")  if is_spike: st.success(f\"**ACCEPTED!** The gain outweighs the cost (F < 0).\") else: st.error(f\"**REJECTED.** Not enough value to justify the cost (F >= 0).\")  with col3: st.subheader(\"Proposed Update\") fig2, ax2 = plt.subplots(figsize=(4, 4)) # Use consistent layout if nodes overlap pos2 = nx.spring_layout(G_a, seed=42) # Highlight new edges new_edges = [e for e in G_a.edges() if e not in G_b.edges()]  nx.draw(G_a, pos2, with_labels=True, node_color='lightgreen', ax=ax2) nx.draw_networkx_edges(G_a, pos2, edgelist=new_edges, edge_color='r', width=2, ax=ax2) st.pyplot(fig2)  st.markdown(\"---\") st.markdown(\"*To run this locally: `pip install streamlit matplotlib networkx && streamlit run examples/playground.py`*\")"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/public_quick_start.py",
        "name": "public_quick_start.py",
        "size": 556,
        "docstring": "Public API Quick Start Example.  Demonstrates how to use the stable public entry points without importing internal modules at top-level."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/simulate_econophysics.py",
        "name": "simulate_econophysics.py",
        "size": 5518,
        "docstring": "Econophysics Eureka Simulation: Fluid Dynamics -> Financial Markets ===================================================================  This script demonstrates a more complex \"Eureka Moment\": Mapping the structural properties of \"Fluid Turbulence\" (Physics) to \"Market Panic/Crash\" (Economics).  Hypothesis: A financial crash structurally resembles fluid turbulence (vortices/loops of panic). If the AI knows about Turbulence, it should recognize a Crash as a similar phenomenon."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/simulate_eureka.py",
        "name": "simulate_eureka.py",
        "size": 4519,
        "docstring": "Eureka Machine Simulation: Cross-Domain Structural Analogy ==========================================================  This script simulates an AI realizing a valid structural analogy between two completely different domains: \"Astronomy\" (Solar System) and \"Quantum Physics\" (Atom Model).  Step 1: Learn \"Solar System\" structure (Hub-and-Spoke). Step 2: Enrich it to understand 'Sun' is a HUB. Step 3: Encounter \"Rutherford Atom\" structure (Nucleus + Electrons). Step 4: Enrich it to understand 'Nucleus' is a HUB. Step 5: Compare and Trigger \"Eureka!\" (High Insight Score)."
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/test_analogy_spike.py",
        "name": "test_analogy_spike.py",
        "size": 5029,
        "docstring": "Verification Script for Structural Analogy Spike ===============================================  This script demonstrates how finding a structural analogy triggers a \"Eureka\" spike in the geDIG metrics.  Scenario: 1. We have a \"Prototype\" structure (e.g., Solar System: Sun + Planets). 2. We observe a new graph update that forms a similar structure (Atom: Nucleus + Electrons). 3. We expect: - High Structural Similarity - Positive Analogy Bonus - Spike in Insight Score (Eureka!)"
      },
      {
        "exists": true,
        "path": "/Users/4d/Documents/GitHub/InsightSpike-AI/examples/test_local_llm.py",
        "name": "test_local_llm.py",
        "size": 1347,
        "docstring": ""
      }
    ],
    "total_size": 116819
  }
}
```
