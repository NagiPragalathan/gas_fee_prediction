"""Main Gas Fee Prediction Pipeline - Orchestrates all components"""

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import Config, NetworkConfig
from .models import GasFeeRecommendation, NetworkState
from .fast_rules import FastRuleBasedRecommendations
from .cached_ml import CachedMLRecommendations
from .feature_engineer import EthereumFeatureEngineer
from .data_collector import EthereumDataCollector
from .utils import format_gas_recommendation, create_sample_trade_scenarios

class GasFeeCompletePipeline:
    """
    Complete Gas Fee Prediction Pipeline
    
    Combines three approaches:
    1. Ultra-fast rule-based (1-2ms)
    2. Cached ML predictions (10-20ms)
    3. Full ML ensemble (100-200ms)
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        self.config = Config()
        self.network_config = NetworkConfig()
        
        # Initialize components
        self.rule_engine = FastRuleBasedRecommendations()
        self.cached_ml = CachedMLRecommendations()
        self.feature_engineer = EthereumFeatureEngineer()
        self.data_collector = EthereumDataCollector()
        
        # ML Models storage
        self.ml_models = {}
        self.feature_columns = []
        
        # Background processing
        self.background_thread = None
        self.stop_background = False
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0
        
        print("🚀 Gas Fee Prediction Pipeline initialized")
    
    def get_instant_recommendation(self, trade_params: Dict) -> Dict:
        """
        Get instant recommendation using fastest available method
        
        Priority:
        1. Cached ML (if available and valid)
        2. Rule-based (always available)
        
        Target latency: 1-5ms
        """
        start_time = time.time()
        
        # Try cached ML first
        cached_recommendation = self.cached_ml.get_recommendation_fast(trade_params)
        
        if cached_recommendation is not None:
            # Add metadata
            cached_recommendation['latency_ms'] = (time.time() - start_time) * 1000
            cached_recommendation['source'] = 'cached_ml'
            cached_recommendation['timestamp'] = datetime.now().isoformat()
            return cached_recommendation
        
        # Fallback to rule-based
        rule_recommendation = self._get_rule_based_recommendation(trade_params)
        rule_recommendation['latency_ms'] = (time.time() - start_time) * 1000
        rule_recommendation['source'] = 'rule_based'
        rule_recommendation['timestamp'] = datetime.now().isoformat()
        
        return rule_recommendation
    
    def get_comprehensive_recommendation(self, trade_params: Dict) -> Dict:
        """
        Get comprehensive recommendation using ensemble approach
        
        Combines:
        1. Rule-based (fast baseline)
        2. Cached ML (if available) 
        3. Fresh ML prediction (if models loaded)
        
        Target latency: 50-200ms
        """
        start_time = time.time()
        recommendations = {}
        
        # 1. Get rule-based recommendation
        try:
            rule_rec = self._get_rule_based_recommendation(trade_params)
            recommendations['rule_based'] = rule_rec
        except Exception as e:
            print(f"⚠️ Rule-based prediction failed: {e}")
        
        # 2. Get cached ML recommendation
        try:
            cached_rec = self.cached_ml.get_recommendation_fast(trade_params)
            if cached_rec:
                recommendations['cached_ml'] = cached_rec
        except Exception as e:
            print(f"⚠️ Cached ML prediction failed: {e}")
        
        # 3. Get fresh ML prediction (if models available)
        try:
            if self.ml_models and self.feature_columns:
                ml_rec = self._get_ml_recommendation(trade_params)
                if ml_rec:
                    recommendations['fresh_ml'] = ml_rec
        except Exception as e:
            print(f"⚠️ Fresh ML prediction failed: {e}")
        
        # Ensemble the recommendations
        if len(recommendations) > 1:
            final_recommendation = self._ensemble_recommendations(recommendations, trade_params)
            final_recommendation['source'] = 'ensemble'
        elif len(recommendations) == 1:
            final_recommendation = list(recommendations.values())[0]
            final_recommendation['source'] = list(recommendations.keys())[0]
        else:
            # Emergency fallback
            final_recommendation = self._get_emergency_recommendation(trade_params)
            final_recommendation['source'] = 'emergency_fallback'
        
        # Add metadata
        final_recommendation['latency_ms'] = (time.time() - start_time) * 1000
        final_recommendation['timestamp'] = datetime.now().isoformat()
        final_recommendation['recommendations_used'] = list(recommendations.keys())
        
        # Update performance tracking
        self.request_count += 1
        self.total_response_time += final_recommendation['latency_ms']
        
        return final_recommendation
    
    def _get_rule_based_recommendation(self, trade_params: Dict) -> Dict:
        """Get rule-based recommendation"""
        return self.rule_engine.get_complete_recommendation(
            base_fee=trade_params.get('base_fee', 25.0),
            network_util=trade_params.get('network_util', 80.0),
            mempool_size=trade_params.get('mempool_size', 150000),
            trade_size_usd=trade_params.get('trade_size_usd', 1000),
            pool_liquidity_usd=trade_params.get('pool_liquidity_usd', self.config.DEFAULT_POOL_LIQUIDITY),
            volatility_score=trade_params.get('volatility_score', self.config.DEFAULT_VOLATILITY_SCORE),
            user_urgency=trade_params.get('user_urgency', 0.5)
        )
    
    def _get_ml_recommendation(self, trade_params: Dict) -> Optional[Dict]:
        """Get fresh ML recommendation using loaded models"""
        try:
            # Create feature vector
            features_df = self._create_feature_vector(trade_params)
            
            if features_df is None or features_df.empty:
                return None
            
            # Ensure we have the required feature columns
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            if missing_cols:
                print(f"⚠️ Missing feature columns: {missing_cols}")
                return None
            
            # Get predictions for each target and quantile
            predictions = {}
            for target_type in ['gas_fee', 'priority_fee', 'slippage']:
                predictions[target_type] = {}
                for quantile in self.config.QUANTILES:
                    model_key = f"{target_type}_q{quantile}"
                    if model_key in self.ml_models:
                        try:
                            model = self.ml_models[model_key]
                            X = features_df[self.feature_columns].fillna(0)
                            pred = model.predict(X)[0]
                            predictions[target_type][f'q{quantile}'] = pred
                        except Exception as e:
                            print(f"⚠️ Prediction failed for {model_key}: {e}")
            
            # Convert to standard format
            return self._format_ml_predictions(predictions)
            
        except Exception as e:
            print(f"⚠️ ML recommendation failed: {e}")
            return None
    
    def _create_feature_vector(self, trade_params: Dict) -> Optional[pd.DataFrame]:
        """Create feature vector from trade parameters"""
        try:
            # Get current network state
            network_state = self.data_collector.get_current_network_state()
            
            # Combine with trade parameters
            combined_data = {**network_state, **trade_params}
            
            # Create DataFrame
            df = pd.DataFrame([combined_data])
            
            # Engineer features
            features_df = self.feature_engineer.create_all_features(df)
            
            return features_df
            
        except Exception as e:
            print(f"⚠️ Feature creation failed: {e}")
            return None
    
    def _format_ml_predictions(self, predictions: Dict) -> Dict:
        """Format ML predictions to standard recommendation format"""
        return {
            'gas_fees': {
                'slow': predictions.get('gas_fee', {}).get('q0.1', 20),
                'standard': predictions.get('gas_fee', {}).get('q0.5', 30),
                'fast': predictions.get('gas_fee', {}).get('q0.9', 50)
            },
            'priority_fees': {
                'low': predictions.get('priority_fee', {}).get('q0.1', 1),
                'medium': predictions.get('priority_fee', {}).get('q0.5', 2),
                'high': predictions.get('priority_fee', {}).get('q0.9', 5)
            },
            'slippage': {
                'aggressive': predictions.get('slippage', {}).get('q0.1', 0.1),
                'balanced': predictions.get('slippage', {}).get('q0.5', 0.5),
                'conservative': predictions.get('slippage', {}).get('q0.9', 1.0)
            }
        }
    
    def _ensemble_recommendations(self, recommendations: Dict, trade_params: Dict) -> Dict:
        """Ensemble multiple recommendations with intelligent weighting"""
        weights = {
            'rule_based': 0.3,
            'cached_ml': 0.4,
            'fresh_ml': 0.5
        }
        
        # Adjust weights based on availability and confidence
        available_methods = list(recommendations.keys())
        total_weight = sum(weights[method] for method in available_methods)
        
        # Normalize weights
        normalized_weights = {method: weights[method] / total_weight for method in available_methods}
        
        # Ensemble each component
        ensembled = {
            'gas_fees': {},
            'priority_fees': {},
            'slippage': {}
        }
        
        for component in ['gas_fees', 'priority_fees', 'slippage']:
            for level in ['slow', 'standard', 'fast', 'low', 'medium', 'high', 'aggressive', 'balanced', 'conservative']:
                weighted_sum = 0
                total_weight = 0
                
                for method, rec in recommendations.items():
                    if component in rec and level in rec[component]:
                        value = rec[component][level]
                        weight = normalized_weights[method]
                        weighted_sum += value * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensembled[component][level] = weighted_sum / total_weight
        
        # Clean up empty levels
        for component in list(ensembled.keys()):
            if not ensembled[component]:
                del ensembled[component]
            else:
                ensembled[component] = {k: v for k, v in ensembled[component].items() if v is not None}
        
        return ensembled
    
    def _get_emergency_recommendation(self, trade_params: Dict) -> Dict:
        """Emergency fallback with safe default values"""
        base_fee = trade_params.get('base_fee', 25.0)
        
        return {
            'gas_fees': {
                'slow': base_fee * 0.9,
                'standard': base_fee * 1.2,
                'fast': base_fee * 1.5
            },
            'priority_fees': {
                'low': 1.0,
                'medium': 2.0,
                'high': 4.0
            },
            'slippage': {
                'aggressive': 0.1,
                'balanced': 0.5,
                'conservative': 1.0
            }
        }
    
    def start_background_ml(self):
        """Start background ML processing thread"""
        if self.background_thread is not None:
            print("⚠️ Background ML already running")
            return
        
        self.stop_background = False
        self.background_thread = threading.Thread(target=self._background_ml_loop, daemon=True)
        self.background_thread.start()
        print("🔄 Background ML processing started")
    
    def stop_background_ml(self):
        """Stop background ML processing"""
        if self.background_thread is None:
            return
        
        self.stop_background = True
        self.background_thread.join(timeout=5)
        self.background_thread = None
        print("⏹️ Background ML processing stopped")
    
    def _background_ml_loop(self):
        """Background loop for updating ML predictions"""
        while not self.stop_background:
            try:
                # Update cached predictions every 10 seconds
                self._update_cached_predictions()
                time.sleep(self.config.BACKGROUND_UPDATE_INTERVAL)
            except Exception as e:
                print(f"⚠️ Background ML error: {e}")
                time.sleep(30)  # Back off on error
    
    def _update_cached_predictions(self):
        """Update cached ML predictions for common parameter combinations"""
        if not self.ml_models or not self.feature_columns:
            return
        
        # Generate predictions for common scenarios
        common_scenarios = create_sample_trade_scenarios()
        cached_predictions = {}
        
        for scenario in common_scenarios[:10]:  # Limit to prevent overload
            try:
                cache_key = self.cached_ml.quantize_params(scenario['params'])
                prediction = self._get_ml_recommendation(scenario['params'])
                
                if prediction:
                    cached_predictions[cache_key] = prediction
                    
            except Exception as e:
                print(f"⚠️ Failed to cache prediction for scenario: {e}")
        
        # Update cache
        self.cached_ml.update_cache(cached_predictions)
    
    def load_models(self, model_dir: str = "models"):
        """Load pre-trained models"""
        try:
            import pickle
            import os
            
            model_files = []
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            if not model_files:
                print(f"⚠️ No model files found in {model_dir}")
                return False
            
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_models[model_data['name']] = model_data['model']
                    
                    if 'feature_columns' in model_data and not self.feature_columns:
                        self.feature_columns = model_data['feature_columns']
            
            print(f"✅ Loaded {len(self.ml_models)} models")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load models: {e}")
            return False
    
    def start_full_system(self) -> bool:
        """Start the complete system with all components"""
        try:
            # Load models if available
            self.load_models()
            
            # Start background processing
            self.start_background_ml()
            
            print("✅ Full system started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start full system: {e}")
            return False
    
    def stop_full_system(self):
        """Stop all system components"""
        self.stop_background_ml()
        print("⏹️ Full system stopped")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
        
        return {
            'requests_processed': self.request_count,
            'average_response_time_ms': avg_response_time,
            'models_loaded': len(self.ml_models),
            'feature_columns': len(self.feature_columns),
            'background_ml_running': self.background_thread is not None and self.background_thread.is_alive(),
            'cached_predictions': len(self.cached_ml.background_predictions)
        }
    
    def run_performance_test(self, num_requests: int = 1000) -> Dict:
        """Run performance test with multiple scenarios"""
        print(f"🏃 Running performance test with {num_requests:,} requests...")
        
        scenarios = create_sample_trade_scenarios()
        response_times = []
        
        start_time = time.time()
        
        for i in range(num_requests):
            scenario = scenarios[i % len(scenarios)]
            
            request_start = time.time()
            recommendation = self.get_instant_recommendation(scenario['params'])
            request_time = (time.time() - request_start) * 1000
            
            response_times.append(request_time)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1:,} requests...")
        
        total_time = time.time() - start_time
        
        return {
            'total_requests': num_requests,
            'total_time_seconds': total_time,
            'requests_per_second': num_requests / total_time,
            'average_response_time_ms': np.mean(response_times),
            'p50_response_time_ms': np.percentile(response_times, 50),
            'p95_response_time_ms': np.percentile(response_times, 95),
            'p99_response_time_ms': np.percentile(response_times, 99),
            'max_response_time_ms': np.max(response_times),
            'min_response_time_ms': np.min(response_times)
        }