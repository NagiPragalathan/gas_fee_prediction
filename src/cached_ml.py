"""Hybrid cached ML-based recommendations with real-time adjustments"""

import time
from typing import Dict, Optional
from .config import Config
from .models import TradeParameters

class CachedMLRecommendations:
    """
    Hybrid cached ML-based recommendations
    
    ML models run in background every 10 seconds and cache predictions.
    Real-time micro-adjustments are applied for rapidly changing factors.
    """
    
    def __init__(self):
        self.config = Config()
        
        # ML models run in background every 10 seconds
        self.background_predictions = {}
        self.last_update = time.time()
        
        # Real-time adjustment cache
        self.adjustment_cache = {}
        
    def get_recommendation_fast(self, trade_params: Dict) -> Optional[Dict]:
        """
        Get cached ML prediction with real-time adjustments (~0.1ms)
        
        Args:
            trade_params: Dictionary of trade parameters
            
        Returns:
            Adjusted ML prediction or None if cache invalid
        """
        # Check if cache is still valid
        if time.time() - self.last_update > self.config.CACHE_VALIDITY_SECONDS:
            return None
            
        # Get cached ML prediction (0ms if cached)
        base_prediction = self.get_cached_prediction(trade_params)
        
        if base_prediction is None:
            return None
            
        # Apply real-time micro-adjustments (~0.1ms)
        real_time_adjustments = self.calculate_micro_adjustments(trade_params)
        
        # Combine (simple math, ~0.05ms)
        return self.apply_adjustments(base_prediction, real_time_adjustments)
    
    def get_cached_prediction(self, trade_params: Dict) -> Optional[Dict]:
        """
        Quantize parameters to create cache key and retrieve prediction
        
        Args:
            trade_params: Dictionary of trade parameters
            
        Returns:
            Cached prediction or None if not found
        """
        cache_key = self.quantize_params(trade_params)
        return self.background_predictions.get(cache_key)
    
    def calculate_micro_adjustments(self, trade_params: Dict) -> Dict[str, float]:
        """
        Calculate real-time adjustments for rapidly changing factors
        
        Uses partial dependence plots (PDP) approach for incremental adjustments
        based on mempool delta and trend analysis.
        
        Args:
            trade_params: Dictionary of trade parameters
            
        Returns:
            Dictionary of adjustment multipliers
        """
        current_mempool_size = trade_params.get('mempool_size', 0)
        cached_mempool_size = self.adjustment_cache.get('mempool_size', current_mempool_size)
        
        if cached_mempool_size == 0:
            mempool_delta = 0
        else:
            mempool_delta = (current_mempool_size - cached_mempool_size) / cached_mempool_size
        
        # Check for monotonous increase/decrease trend
        mempool_trend = self.adjustment_cache.get('mempool_trend', 0)
        if abs(mempool_delta) > 0.1:  # Significant change threshold
            mempool_trend = mempool_delta
            self.adjustment_cache['mempool_trend'] = mempool_trend
        
        # PDP-based multiplier effects
        return {
            'gas_multiplier': 1 + (mempool_delta * 0.1),
            'priority_multiplier': 1 + (mempool_delta * 0.15),
            'slippage_multiplier': 1 + (abs(mempool_delta) * 0.05),
            'trend_multiplier': 1 + (mempool_trend * 0.05)
        }
    
    def apply_adjustments(self, base_prediction: Dict, adjustments: Dict[str, float]) -> Dict:
        """
        Apply real-time adjustments to cached predictions
        
        Args:
            base_prediction: Base ML prediction from cache
            adjustments: Real-time adjustment multipliers
            
        Returns:
            Adjusted prediction dictionary
        """
        adjusted = base_prediction.copy()
        
        # Apply multipliers to gas fees
        if 'gas_fees' in adjusted:
            gas_mult = adjustments.get('gas_multiplier', 1.0)
            for speed in adjusted['gas_fees']:
                adjusted['gas_fees'][speed] *= gas_mult
        
        # Apply multipliers to priority fees
        if 'priority_fees' in adjusted:
            priority_mult = adjustments.get('priority_multiplier', 1.0)
            for priority in adjusted['priority_fees']:
                adjusted['priority_fees'][priority] *= priority_mult
        
        # Apply multipliers to slippage
        if 'slippage' in adjusted:
            slippage_mult = adjustments.get('slippage_multiplier', 1.0)
            for tolerance in adjusted['slippage']:
                adjusted['slippage'][tolerance] *= slippage_mult
        
        adjusted['source'] = 'cached_ml_adjusted'
        adjusted['adjustments_applied'] = adjustments
        
        return adjusted
    
    def quantize_params(self, trade_params: Dict) -> str:
        """
        Quantize parameters for caching (create cache buckets)
        
        Args:
            trade_params: Dictionary of trade parameters
            
        Returns:
            String cache key
        """
        # Round to nearest cache bucket for efficient caching
        quantized = {
            'base_fee_bucket': round(trade_params.get('base_fee', 0), 1),
            'util_bucket': round(trade_params.get('network_util', 0) / 5) * 5,
            'mempool_bucket': round(trade_params.get('mempool_size', 0) / 10000) * 10000,
            'trade_size_bucket': round(trade_params.get('trade_size_usd', 0) / 1000) * 1000
        }
        return str(quantized)
    
    def update_cache(self, predictions: Dict):
        """
        Update cached predictions from background ML training
        
        Args:
            predictions: New predictions to cache
        """
        self.background_predictions.update(predictions)
        self.last_update = time.time()
    
    def clear_cache(self):
        """Clear all cached predictions"""
        self.background_predictions.clear()
        self.adjustment_cache.clear()
        self.last_update = time.time()
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_predictions': len(self.background_predictions),
            'last_update': self.last_update,
            'cache_age_seconds': time.time() - self.last_update,
            'is_valid': time.time() - self.last_update <= self.config.CACHE_VALIDITY_SECONDS
        } 