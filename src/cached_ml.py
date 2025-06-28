"""Hybrid cached ML-based recommendations with real-time adjustments"""

import time
import threading
import pickle
import os
from pathlib import Path
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
        
        # Background thread management
        self.background_running = False
        self.background_thread = None
        
        # Add cache persistence
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "ml_predictions.pkl"
        
        # Load existing cache
        self._load_cache_from_disk()
        
    def start_background_updates(self):
        """Start background ML updates with immediate initial population"""
        if self.background_running:
            return
            
        print("🔄 Starting background ML updates...")
        
        # Generate initial cache immediately
        try:
            print("🚀 Generating initial ML cache...")
            self._generate_background_predictions()
            print(f"✅ Initial cache ready with {len(self.background_predictions)} predictions")
        except Exception as e:
            print(f"❌ Initial cache generation failed: {e}")
        
        # Start background thread
        self.background_running = True
        self.background_thread = threading.Thread(target=self._background_ml_loop, daemon=True)
        self.background_thread.start()
        print("✅ Background ML thread started")
    
    def stop_background_updates(self):
        """Stop background ML updates"""
        self.background_running = False
        if self.background_thread:
            self.background_thread.join(timeout=1)
        print("⏹️ Stopped background ML updates")
    
    def update_real_time_cache(self, network_data: Dict):
        """Update real-time cache with fresh network data"""
        if not network_data:
            return
            
        # Update adjustment cache with current network state
        self.adjustment_cache.update({
            'mempool_size': network_data.get('mempool_pending_count', 0),
            'network_util': network_data.get('network_utilization', 0),
            'base_fee': network_data.get('baseFeePerGas', 0) / 1e9,
            'timestamp': time.time()
        })
    
    def _background_ml_loop(self):
        """Background loop for ML model updates"""
        while self.background_running:
            try:
                # Simulate ML model training and prediction generation
                self._generate_background_predictions()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                print(f"❌ Background ML error: {e}")
                time.sleep(5)
    
    def _generate_background_predictions(self):
        """Generate background ML predictions for common parameter ranges"""
        # Generate predictions for common parameter combinations
        predictions = {}
        
        for base_fee in [10, 15, 20, 25, 30, 40, 50, 75, 100]:
            for util in [30, 50, 70, 80, 90, 95]:
                for mempool in [50000, 100000, 150000, 200000, 300000]:
                    for trade_size in [1000, 5000, 10000, 50000]:
                        trade_params = {
                            'base_fee': base_fee,
                            'network_util': util,
                            'mempool_size': mempool,
                            'trade_size_usd': trade_size
                        }
                        
                        cache_key = self.quantize_params(trade_params)
                        predictions[cache_key] = self._simulate_ml_prediction(trade_params)
        
        # Update cache with new predictions
        self.update_cache(predictions)
    
    def _simulate_ml_prediction(self, trade_params: Dict) -> Dict:
        """Simulate ML model prediction (placeholder for real ML)"""
        base_fee = trade_params.get('base_fee', 25)
        network_util = trade_params.get('network_util', 80)
        mempool_size = trade_params.get('mempool_size', 150000)
        
        # Simulate sophisticated ML prediction
        congestion_factor = 1 + (network_util / 100) * 0.5
        mempool_factor = 1 + (mempool_size / 100000) * 0.2
        
        ml_multiplier = congestion_factor * mempool_factor
        
        return {
            'gas_fees': {
                'slow': base_fee * 0.8 * ml_multiplier,
                'standard': base_fee * 1.0 * ml_multiplier,
                'fast': base_fee * 1.3 * ml_multiplier,
                'rapid': base_fee * 1.6 * ml_multiplier
            },
            'priority_fees': {
                'low': 1.5 * ml_multiplier,
                'medium': 2.5 * ml_multiplier,
                'high': 4.0 * ml_multiplier,
                'urgent': 6.0 * ml_multiplier
            },
            'slippage': {
                'conservative': 0.5 * ml_multiplier,
                'balanced': 0.3 * ml_multiplier,
                'aggressive': 0.1 * ml_multiplier
            },
            'confidence': 0.85,
            'source': 'cached_ml',
            'generated_at': time.time()
        }
    
    def get_recommendation_fast(self, trade_params: Dict) -> Optional[Dict]:
        """
        Get cached ML prediction with real-time adjustments (~0.1ms)
        """
        # Get cached ML prediction first
        base_prediction = self.get_cached_prediction(trade_params)
        
        if base_prediction is None:
            # If no exact match, try to generate a quick prediction
            return self._generate_fallback_prediction(trade_params)
        
        # Check if cache is still reasonably fresh (more lenient)
        cache_age = time.time() - self.last_update
        if cache_age > self.config.CACHE_VALIDITY_SECONDS:
            print(f"⚠️ Cache is {cache_age:.0f}s old, using with caution")
            # Still use the cache but mark it as potentially stale
            base_prediction['cache_age'] = cache_age
            base_prediction['confidence'] = max(0.5, base_prediction.get('confidence', 0.85) - 0.1)
        
        # Apply real-time micro-adjustments
        real_time_adjustments = self.calculate_micro_adjustments(trade_params)
        return self.apply_adjustments(base_prediction, real_time_adjustments)
    
    def _generate_fallback_prediction(self, trade_params: Dict) -> Dict:
        """Generate a quick ML-style prediction when cache is empty"""
        print("🔄 Generating fallback prediction (cache building...)")
        
        # Use the same logic as the background ML but for current params
        base_prediction = self._simulate_ml_prediction(trade_params)
        base_prediction['source'] = 'cached_ml_fallback'
        base_prediction['confidence'] = 0.7  # Lower confidence for fallback
        
        # Apply real-time adjustments
        real_time_adjustments = self.calculate_micro_adjustments(trade_params)
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
        """Update cached predictions and save to disk"""
        self.background_predictions.update(predictions)
        self.last_update = time.time()
        
        # Save to disk for persistence
        self._save_cache_to_disk()
    
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
            'is_valid': time.time() - self.last_update <= self.config.CACHE_VALIDITY_SECONDS,
            'background_running': self.background_running
        }
    
    def _load_cache_from_disk(self):
        """Load cache from disk if available"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.background_predictions = cache_data.get('predictions', {})
                    self.last_update = cache_data.get('last_update', time.time())
                print(f"📂 Loaded {len(self.background_predictions)} cached predictions")
            except Exception as e:
                print(f"❌ Cache load error: {e}")
    
    def _save_cache_to_disk(self):
        """Save cache to disk"""
        try:
            cache_data = {
                'predictions': self.background_predictions,
                'last_update': self.last_update,
                'version': '1.0'
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Saved {len(self.background_predictions)} predictions to disk")
        except Exception as e:
            print(f"❌ Cache save error: {e}") 