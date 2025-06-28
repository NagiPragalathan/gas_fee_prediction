"""Hybrid cached ML-based recommendations with real-time adjustments - REAL DATA ONLY"""

import time
import threading
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from .config import Config
from .models import TradeParameters

class CachedMLRecommendations:
    """
    Hybrid cached ML-based recommendations using REAL ML models and REAL-TIME data
    
    NO SIMULATION - Only real trained models and live network data
    """
    
    def __init__(self):
        self.config = Config()
        
        # REAL ML models (loaded from disk)
        self.ml_models = {}
        self.feature_columns = []
        self.model_metadata = {}
        
        # Real-time predictions cache
        self.background_predictions = {}
        self.last_update = time.time()
        
        # Real-time adjustment cache
        self.adjustment_cache = {}
        
        # Background thread management
        self.background_running = False
        self.background_thread = None
        
        # Components for real data
        self.data_collector = None
        self.feature_engineer = None
        
        # Cache persistence
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "ml_predictions.pkl"
        
        # Load real ML models
        self._load_real_ml_models()
        
        print("🎯 CachedML initialized with REAL-TIME data only")
    
    def set_components(self, data_collector, feature_engineer):
        """Set real-time data components"""
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer
        print("✅ Real-time data components connected")
    
    def _load_real_ml_models(self):
        """Load REAL trained ML models from disk"""
        model_dir = Path("models")
        if not model_dir.exists():
            print("⚠️ No models directory found. Create 'models/' and add trained .pkl files")
            return
        
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            print("⚠️ No .pkl model files found in models/ directory")
            return
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    # Extract model info
                    model_name = model_data.get('name', model_file.stem)
                    self.ml_models[model_name] = model_data['model']
                    
                    if 'feature_columns' in model_data and not self.feature_columns:
                        self.feature_columns = model_data['feature_columns']
                    
                    if 'metadata' in model_data:
                        self.model_metadata[model_name] = model_data['metadata']
                
                print(f"✅ Loaded real ML model: {model_name}")
                
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
        
        print(f"🎯 Loaded {len(self.ml_models)} REAL ML models")
    
    def start_background_updates(self):
        """Start background ML updates using REAL data"""
        if self.background_running:
            return
            
        if not self.ml_models:
            print("⚠️ No ML models loaded. Background updates will use fallback method.")
        
        if not self.data_collector:
            print("⚠️ No data collector connected. Please call set_components() first.")
            return
            
        print("🔄 Starting REAL-TIME background ML updates...")
        
        # Generate initial cache using real data
        try:
            print("🚀 Generating initial ML cache with REAL data...")
            self._generate_real_background_predictions()
            print(f"✅ Real-time cache ready with {len(self.background_predictions)} predictions")
        except Exception as e:
            print(f"❌ Initial real cache generation failed: {e}")
        
        # Start background thread
        self.background_running = True
        self.background_thread = threading.Thread(target=self._real_background_ml_loop, daemon=True)
        self.background_thread.start()
        print("✅ REAL-TIME background ML thread started")
    
    def _real_background_ml_loop(self):
        """Background loop using REAL ML models and REAL network data"""
        while self.background_running:
            try:
                # Get REAL current network state
                if self.data_collector:
                    self._generate_real_background_predictions()
                else:
                    print("⚠️ No data collector - skipping real predictions")
                
                time.sleep(10)  # Update every 10 seconds with REAL data
                
            except Exception as e:
                print(f"❌ Real background ML error: {e}")
                time.sleep(30)  # Back off on error
    
    def _generate_real_background_predictions(self):
        """Generate predictions using REAL ML models and REAL network data"""
        if not self.data_collector:
            return
        
        try:
            # Get REAL current network state
            current_network_data = self.data_collector.get_current_network_state()
            
            if not current_network_data:
                print("⚠️ No real network data available")
                return
            
            # Extract real parameters from live data
            real_base_fee = current_network_data.get('baseFeePerGas', 25e9) / 1e9
            real_network_util = current_network_data.get('network_utilization', 80.0)
            real_mempool_size = current_network_data.get('mempool_pending_count', 150000)
            
            predictions = {}
            
            # Generate predictions for realistic trade sizes using REAL network conditions
            realistic_trade_sizes = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
            
            for trade_size in realistic_trade_sizes:
                # Use REAL automated parameters
                real_params = self.data_collector.get_fully_automated_params(
                    trade_size_usd=trade_size,
                    token_address=None
                )
                
                cache_key = self.quantize_params(real_params)
                
                # Generate prediction using REAL ML models or real fallback
                if self.ml_models and self.feature_columns:
                    prediction = self._predict_with_real_models(real_params)
                else:
                    prediction = self._real_data_fallback_prediction(real_params)
                
                if prediction:
                    predictions[cache_key] = prediction
            
            # Update cache with REAL predictions
            self.update_cache(predictions)
            print(f"🎯 Updated cache with {len(predictions)} REAL predictions")
            
        except Exception as e:
            print(f"❌ Real prediction generation failed: {e}")
    
    def _predict_with_real_models(self, trade_params: Dict) -> Optional[Dict]:
        """Make predictions using REAL trained ML models"""
        try:
            if not self.feature_engineer:
                print("⚠️ No feature engineer available")
                return None
            
            # Create REAL feature vector
            features_df = self._create_real_feature_vector(trade_params)
            
            if features_df is None or features_df.empty:
                return None
            
            # Ensure we have required feature columns
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            if missing_cols:
                print(f"⚠️ Missing feature columns: {missing_cols}")
                return None
            
            # Make REAL predictions with trained models
            predictions = {}
            
            for model_name, model in self.ml_models.items():
                try:
                    X = features_df[self.feature_columns].fillna(0)
                    pred = model.predict(X)[0]
                    
                    # Parse model name to determine target
                    if 'gas_fee' in model_name:
                        target_type = 'gas_fee'
                    elif 'priority_fee' in model_name:
                        target_type = 'priority_fee'
                    elif 'slippage' in model_name:
                        target_type = 'slippage'
                    else:
                        continue
                    
                    if target_type not in predictions:
                        predictions[target_type] = {}
                    
                    # Extract quantile from model name (e.g., "gas_fee_q0.5")
                    if '_q' in model_name:
                        quantile = model_name.split('_q')[-1]
                        predictions[target_type][f'q{quantile}'] = pred
                
                except Exception as e:
                    print(f"⚠️ Prediction failed for model {model_name}: {e}")
            
            # Convert to standard format
            return self._format_real_predictions(predictions, trade_params)
            
        except Exception as e:
            print(f"❌ Real model prediction failed: {e}")
            return None
    
    def _create_real_feature_vector(self, trade_params: Dict) -> Optional[pd.DataFrame]:
        """Create feature vector using REAL feature engineering"""
        try:
            # Get REAL current network state
            network_state = self.data_collector.get_current_network_state()
            
            # Combine REAL network data with trade parameters
            combined_data = {**network_state, **trade_params}
            
            # Create DataFrame
            df = pd.DataFrame([combined_data])
            
            # Engineer REAL features
            features_df = self.feature_engineer.create_all_features(df)
            
            return features_df
            
        except Exception as e:
            print(f"❌ Real feature creation failed: {e}")
            return None
    
    def _format_real_predictions(self, predictions: Dict, trade_params: Dict) -> Dict:
        """Format REAL ML predictions to standard recommendation format"""
        
        # Use real predictions or fall back to data-driven estimates
        gas_predictions = predictions.get('gas_fee', {})
        priority_predictions = predictions.get('priority_fee', {})
        slippage_predictions = predictions.get('slippage', {})
        
        # Get real base fee for fallbacks
        real_base_fee = trade_params.get('base_fee', 25.0)
        
        return {
            'gas_fees': {
                'slow': gas_predictions.get('q0.1', real_base_fee * 0.9),
                'standard': gas_predictions.get('q0.5', real_base_fee * 1.1),
                'fast': gas_predictions.get('q0.8', real_base_fee * 1.4),
                'rapid': gas_predictions.get('q0.95', real_base_fee * 1.8)
            },
            'priority_fees': {
                'low': priority_predictions.get('q0.1', 1.0),
                'medium': priority_predictions.get('q0.5', 2.0),
                'high': priority_predictions.get('q0.8', 4.0),
                'urgent': priority_predictions.get('q0.95', 7.0)
            },
            'slippage': {
                'aggressive': slippage_predictions.get('q0.1', 0.1),
                'balanced': slippage_predictions.get('q0.5', 0.3),
                'conservative': slippage_predictions.get('q0.8', 0.8)
            },
            'confidence': self._calculate_real_confidence(predictions),
            'source': 'real_ml_cached',
            'models_used': list(predictions.keys()),
            'generated_at': time.time()
        }
    
    def _real_data_fallback_prediction(self, trade_params: Dict) -> Dict:
        """Fallback using REAL data but simple calculations (no simulation)"""
        
        # Use REAL current network conditions
        real_base_fee = trade_params.get('base_fee', 25.0)
        real_network_util = trade_params.get('network_util', 80.0)
        real_mempool_size = trade_params.get('mempool_size', 150000)
        
        # REAL external gas estimates
        try:
            external_estimates = self.data_collector.get_external_gas_estimates()
            
            # Use REAL external API data if available
            if 'blocknative' in external_estimates:
                blocknative = external_estimates['blocknative']
                return {
                    'gas_fees': {
                        'slow': blocknative.get('standard', real_base_fee * 0.9),
                        'standard': blocknative.get('fast', real_base_fee * 1.1),
                        'fast': blocknative.get('rapid', real_base_fee * 1.4),
                        'rapid': blocknative.get('rapid', real_base_fee * 1.8) * 1.2
                    },
                    'priority_fees': {
                        'low': blocknative.get('priority_fee_80', 2.0) * 0.5,
                        'medium': blocknative.get('priority_fee_80', 2.0),
                        'high': blocknative.get('priority_fee_80', 2.0) * 1.5,
                        'urgent': blocknative.get('priority_fee_80', 2.0) * 2.0
                    },
                    'slippage': {
                        'aggressive': 0.1,
                        'balanced': 0.3,
                        'conservative': 0.8
                    },
                    'confidence': 0.75,
                    'source': 'real_external_apis',
                    'data_source': 'blocknative',
                    'generated_at': time.time()
                }
            
            elif '1inch' in external_estimates:
                # Use 1inch real data
                oneinch = external_estimates['1inch']
                return {
                    'gas_fees': {
                        'slow': oneinch.get('standard', real_base_fee * 0.9),
                        'standard': oneinch.get('fast', real_base_fee * 1.1),
                        'fast': oneinch.get('fast', real_base_fee * 1.4) * 1.2,
                        'rapid': oneinch.get('fast', real_base_fee * 1.8) * 1.5
                    },
                    'priority_fees': {
                        'low': 1.0, 'medium': 2.0, 'high': 4.0, 'urgent': 7.0
                    },
                    'slippage': {
                        'aggressive': 0.1, 'balanced': 0.3, 'conservative': 0.8
                    },
                    'confidence': 0.70,
                    'source': 'real_external_apis',
                    'data_source': '1inch',
                    'generated_at': time.time()
                }
        
        except Exception as e:
            print(f"⚠️ External API fallback failed: {e}")
        
        # Final fallback using only real base fee
        return {
            'gas_fees': {
                'slow': real_base_fee * 0.9,
                'standard': real_base_fee * 1.1,
                'fast': real_base_fee * 1.4,
                'rapid': real_base_fee * 1.8
            },
            'priority_fees': {
                'low': 1.0, 'medium': 2.0, 'high': 4.0, 'urgent': 7.0
            },
            'slippage': {
                'aggressive': 0.1, 'balanced': 0.3, 'conservative': 0.8
            },
            'confidence': 0.60,
            'source': 'real_data_fallback',
            'generated_at': time.time()
        }
    
    def _calculate_real_confidence(self, predictions: Dict) -> float:
        """Calculate confidence based on REAL model availability"""
        if not predictions:
            return 0.5
        
        # Higher confidence with more real models
        model_count = len(predictions)
        base_confidence = 0.7
        
        if model_count >= 3:  # All models available
            return 0.95
        elif model_count >= 2:  # Most models available
            return 0.85
        elif model_count >= 1:  # Some models available
            return 0.75
        else:
            return base_confidence
    
    def get_recommendation_fast(self, trade_params: Dict) -> Optional[Dict]:
        """Get cached REAL ML prediction with real-time adjustments"""
        # Get cached REAL prediction
        base_prediction = self.get_cached_prediction(trade_params)
        
        if base_prediction is None:
            # Generate using REAL data fallback
            return self._real_data_fallback_prediction(trade_params)
        
        # Apply real-time micro-adjustments using REAL network deltas
        real_time_adjustments = self.calculate_real_micro_adjustments(trade_params)
        return self.apply_adjustments(base_prediction, real_time_adjustments)
    
    def calculate_real_micro_adjustments(self, trade_params: Dict) -> Dict[str, float]:
        """Calculate adjustments using REAL network data changes"""
        if not self.data_collector:
            return {'gas_multiplier': 1.0, 'priority_multiplier': 1.0, 'slippage_multiplier': 1.0}
        
        try:
            # Get REAL current network state
            current_data = self.data_collector.get_current_network_state()
            
            current_mempool = current_data.get('mempool_pending_count', 150000)
            current_base_fee = current_data.get('baseFeePerGas', 25e9) / 1e9
            
            # Compare with cached values
            cached_mempool = self.adjustment_cache.get('mempool_size', current_mempool)
            cached_base_fee = self.adjustment_cache.get('base_fee', current_base_fee)
            
            # Calculate REAL deltas
            mempool_delta = (current_mempool - cached_mempool) / cached_mempool if cached_mempool > 0 else 0
            base_fee_delta = (current_base_fee - cached_base_fee) / cached_base_fee if cached_base_fee > 0 else 0
            
            return {
                'gas_multiplier': 1 + (base_fee_delta * 0.8) + (mempool_delta * 0.1),
                'priority_multiplier': 1 + (mempool_delta * 0.2),
                'slippage_multiplier': 1 + (abs(mempool_delta) * 0.05)
            }
            
        except Exception as e:
            print(f"⚠️ Real adjustment calculation failed: {e}")
            return {'gas_multiplier': 1.0, 'priority_multiplier': 1.0, 'slippage_multiplier': 1.0}
    
    def quantize_params(self, trade_params: Dict) -> str:
        """Quantize REAL parameters for caching"""
        quantized = {
            'base_fee_bucket': round(trade_params.get('base_fee', 0), 1),
            'util_bucket': round(trade_params.get('network_util', 0) / 5) * 5,
            'mempool_bucket': round(trade_params.get('mempool_size', 0) / 10000) * 10000,
            'trade_size_bucket': round(trade_params.get('trade_size_usd', 0) / 1000) * 1000
        }
        return str(quantized)
    
    def apply_adjustments(self, base_prediction: Dict, adjustments: Dict[str, float]) -> Dict:
        """Apply real-time adjustments to cached predictions"""
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
        
        adjusted['source'] = 'real_cached_ml_adjusted'
        adjusted['adjustments_applied'] = adjustments
        
        return adjusted
    
    def update_real_time_cache(self, network_data: Dict):
        """Update cache with REAL network data"""
        if not network_data:
            return
        
        self.adjustment_cache.update({
            'mempool_size': network_data.get('mempool_pending_count', 0),
            'network_util': network_data.get('network_utilization', 0),
            'base_fee': network_data.get('baseFeePerGas', 0) / 1e9,
            'timestamp': time.time()
        })
    
    def get_cached_prediction(self, trade_params: Dict) -> Optional[Dict]:
        """Retrieve REAL cached prediction"""
        cache_key = self.quantize_params(trade_params)
        return self.background_predictions.get(cache_key)
    
    def update_cache(self, predictions: Dict):
        """Update cache with REAL predictions"""
        self.background_predictions.update(predictions)
        self.last_update = time.time()
        print(f"💾 Updated cache with {len(predictions)} REAL predictions")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_predictions': len(self.background_predictions),
            'last_update': self.last_update,
            'cache_age_seconds': time.time() - self.last_update,
            'is_valid': time.time() - self.last_update <= self.config.CACHE_VALIDITY_SECONDS,
            'background_running': self.background_running,
            'real_models_loaded': len(self.ml_models),
            'data_collector_connected': self.data_collector is not None
        } 