"""Data models and schemas for Gas Fee Prediction System"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class PredictionSource(Enum):
    """Source of gas fee prediction"""
    RULE_BASED = "rule_based"
    CACHED_ML = "cached_ml"
    ML_PREDICTION = "ml_prediction" 
    ENSEMBLE = "ensemble"
    DEFAULT_FALLBACK = "default_fallback"

class ConfidenceLevel(Enum):
    """Confidence level of prediction"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class NetworkState:
    """Current Ethereum network state"""
    base_fee_per_gas: int  # wei
    gas_used: int
    gas_limit: int
    network_utilization: float  # percentage
    block_number: int
    timestamp: datetime
    mempool_pending_count: int
    mempool_total_size: int
    median_priority_fee: float  # gwei
    avg_slippage: float

@dataclass
class TradeParameters:
    """Parameters for a specific trade"""
    base_fee: float  # gwei
    network_util: float  # percentage
    mempool_size: int
    trade_size_usd: float
    pool_liquidity_usd: float
    volatility_score: float  # 0-1
    user_urgency: float  # 0-1
    mempool_congestion: Optional[float] = None

@dataclass
class GasFeeRecommendation:
    """Gas fee recommendation with multiple speed options"""
    slow: float
    standard: float
    fast: float
    rapid: Optional[float] = None

@dataclass
class PriorityFeeRecommendation:
    """Priority fee recommendation with different urgency levels"""
    low: float
    medium: float
    high: float
    urgent: Optional[float] = None

@dataclass
class SlippageRecommendation:
    """Slippage tolerance recommendation"""
    conservative: float
    balanced: float
    aggressive: float

@dataclass
class PredictionResult:
    """Complete prediction result"""
    gas_fees: GasFeeRecommendation
    priority_fees: PriorityFeeRecommendation
    slippage: SlippageRecommendation
    source: PredictionSource
    confidence: ConfidenceLevel
    latency_ms: float
    timestamp: str
    network_state: Optional[Dict] = None
    trade_parameters: Optional[Dict] = None
    sources_used: Optional[List[str]] = None

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    total_requests: int

@dataclass
class FeatureImportance:
    """Feature importance from ML models"""
    feature_name: str
    importance: float
    category: str 