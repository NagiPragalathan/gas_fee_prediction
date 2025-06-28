"""
Test script for newly added features:
- Network Health Features (5 features)
- Miner/Validator Features (5 features) 
- Transaction Type Features (5 features)
- Enhanced data collection methods
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import EthereumDataCollector
from src.feature_engineer import EthereumFeatureEngineer
from src.pipeline import GasFeeCompletePipeline

def test_enhanced_data_collection():
    """Test the enhanced data collection methods"""
    print("🔍 === ENHANCED DATA COLLECTION TEST ===")
    
    collector = EthereumDataCollector()
    
    print("1. Testing enhanced network state...")
    try:
        enhanced_state = collector.get_enhanced_network_state()
        print(f"✅ Enhanced network state collected:")
        print(f"   - Basic data keys: {list(enhanced_state.keys())}")
        print(f"   - Block number: {enhanced_state.get('blockNumber', 'N/A')}")
        print(f"   - Network utilization: {enhanced_state.get('network_utilization', 'N/A'):.2f}%")
        
        if 'mempool_detailed' in enhanced_state:
            print(f"   - Mempool analysis: {enhanced_state['mempool_detailed']}")
        if 'block_timing_stats' in enhanced_state:
            print(f"   - Block timing: {enhanced_state['block_timing_stats']}")
        if 'tx_type_estimates' in enhanced_state:
            print(f"   - TX type estimates: {enhanced_state['tx_type_estimates']}")
            
    except Exception as e:
        print(f"❌ Enhanced network state failed: {e}")
    
    print("\n2. Testing real uncle block detection...")
    try:
        uncle_rate = collector._get_real_uncle_blocks()
        print(f"✅ Uncle block rate: {uncle_rate:.4f} ({uncle_rate*100:.2f}%)")
    except Exception as e:
        print(f"❌ Uncle block detection failed: {e}")
    
    print("\n3. Testing reorg detection...")
    try:
        reorg_freq = collector._detect_real_reorgs()
        print(f"✅ Reorg frequency: {reorg_freq:.6f}")
    except Exception as e:
        print(f"❌ Reorg detection failed: {e}")
    
    print("\n4. Testing MEV/Flashbots data...")
    try:
        flashbots_data = collector._get_real_flashbots_data()
        print(f"✅ Flashbots data: {flashbots_data}")
    except Exception as e:
        print(f"❌ Flashbots data failed: {e}")
    
    print("\n5. Testing validator participation data...")
    try:
        validator_data = collector._get_real_validator_data()
        print(f"✅ Validator data: {validator_data}")
    except Exception as e:
        print(f"❌ Validator data failed: {e}")
    
    print("\n" + "="*60 + "\n")

def test_new_feature_engineering():
    """Test the newly added feature engineering methods"""
    print("🛠️ === NEW FEATURE ENGINEERING TEST ===")
    
    # Create sample data
    collector = EthereumDataCollector()
    engineer = EthereumFeatureEngineer()
    
    print("1. Collecting real network data for feature engineering...")
    try:
        network_state = collector.get_current_network_state()
        print(f"✅ Network data collected: Block {network_state['blockNumber']}")
    except Exception as e:
        print(f"❌ Network data collection failed: {e}")
        # Use mock data
        network_state = {
            'baseFeePerGas': 25000000000,  # 25 gwei
            'gasUsed': 27000000,
            'gasLimit': 30000000,
            'network_utilization': 90.0,
            'blockNumber': 18500000,
            'timestamp': datetime.now(),
            'mempool_pending_count': 180000,
            'mempool_total_size': 75000000,
            'median_priority_fee': 3.5
        }
    
    # Create DataFrame
    df = pd.DataFrame([network_state])
    
    print("\n2. Testing Network Health Features...")
    try:
        df_health = engineer.add_network_health_features(df.copy())
        health_features = [
            'uncle_block_rate', 'reorg_frequency', 'node_sync_health',
            'validator_participation', 'finalization_delay'
        ]
        
        print("✅ Network Health Features:")
        for feature in health_features:
            if feature in df_health.columns:
                value = df_health[feature].iloc[0]
                print(f"   - {feature}: {value:.6f}")
            else:
                print(f"   - {feature}: ❌ MISSING")
                
    except Exception as e:
        print(f"❌ Network Health Features failed: {e}")
    
    print("\n3. Testing Miner/Validator Features...")
    try:
        df_miner = engineer.add_miner_validator_features(df.copy())
        miner_features = [
            'miner_revenue_per_block', 'fee_revenue_ratio', 'miner_base_fee_preference',
            'flashbots_bundle_ratio', 'private_mempool_ratio'
        ]
        
        print("✅ Miner/Validator Features:")
        for feature in miner_features:
            if feature in df_miner.columns:
                value = df_miner[feature].iloc[0]
                print(f"   - {feature}: {value:.6f}")
            else:
                print(f"   - {feature}: ❌ MISSING")
                
    except Exception as e:
        print(f"❌ Miner/Validator Features failed: {e}")
    
    print("\n4. Testing Transaction Type Features...")
    try:
        df_tx = engineer.add_transaction_type_features(df.copy())
        tx_features = [
            'simple_transfer_ratio', 'complex_contract_ratio', 'failed_transaction_ratio',
            'gas_intensive_tx_ratio', 'average_tx_gas_used'
        ]
        
        print("✅ Transaction Type Features:")
        for feature in tx_features:
            if feature in df_tx.columns:
                value = df_tx[feature].iloc[0]
                print(f"   - {feature}: {value:.6f}")
            else:
                print(f"   - {feature}: ❌ MISSING")
                
    except Exception as e:
        print(f"❌ Transaction Type Features failed: {e}")
    
    print("\n5. Testing Complete Feature Engineering...")
    try:
        df_complete = engineer.create_all_features(df.copy())
        total_features = len(df_complete.columns)
        new_features = total_features - len(df.columns)
        
        print(f"✅ Complete Feature Engineering:")
        print(f"   - Original columns: {len(df.columns)}")
        print(f"   - Total features created: {total_features}")
        print(f"   - New features added: {new_features}")
        print(f"   - Expected total: {engineer.get_total_feature_count()}")
        
        # Show feature categories
        categories = engineer.get_feature_categories()
        print(f"\n   Feature Categories:")
        for category, features in categories.items():
            present_features = [f for f in features if f in df_complete.columns]
            print(f"   - {category}: {len(present_features)}/{len(features)} features")
            if len(present_features) != len(features):
                missing = [f for f in features if f not in df_complete.columns]
                print(f"     Missing: {missing}")
        
    except Exception as e:
        print(f"❌ Complete Feature Engineering failed: {e}")
    
    print("\n" + "="*60 + "\n")

def test_pipeline_integration():
    """Test integration with the complete pipeline"""
    print("🚀 === PIPELINE INTEGRATION TEST ===")
    
    print("1. Initializing complete pipeline...")
    try:
        pipeline = GasFeeCompletePipeline()
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return
    
    print("\n2. Testing instant recommendation with new features...")
    try:
        trade_params = {
            'base_fee': 25.0,
            'network_util': 85.0,
            'mempool_size': 180000,
            'trade_size_usd': 5000,
            'pool_liquidity_usd': 1500000,
            'volatility_score': 0.7,
            'user_urgency': 0.8
        }
        
        start_time = time.time()
        recommendation = pipeline.get_instant_recommendation(trade_params)
        response_time = (time.time() - start_time) * 1000
        
        print(f"✅ Instant recommendation generated in {response_time:.2f}ms")
        print(f"   - Source: {recommendation.get('source', 'unknown')}")
        print(f"   - Gas fees: {recommendation.get('gas_fees', {})}")
        print(f"   - Priority fees: {recommendation.get('priority_fees', {})}")
        
    except Exception as e:
        print(f"❌ Instant recommendation failed: {e}")
    
    print("\n3. Testing comprehensive recommendation...")
    try:
        start_time = time.time()
        comprehensive = pipeline.get_comprehensive_recommendation(trade_params)
        response_time = (time.time() - start_time) * 1000
        
        print(f"✅ Comprehensive recommendation generated in {response_time:.2f}ms")
        print(f"   - Source: {comprehensive.get('source', 'unknown')}")
        print(f"   - Sources used: {comprehensive.get('recommendations_used', [])}")
        
    except Exception as e:
        print(f"❌ Comprehensive recommendation failed: {e}")
    
    print("\n4. Testing system status...")
    try:
        status = pipeline.get_system_status()
        print(f"✅ System status:")
        print(f"   - Requests processed: {status.get('requests_processed', 0)}")
        print(f"   - Average response time: {status.get('average_response_time_ms', 0):.2f}ms")
        print(f"   - Models loaded: {status.get('models_loaded', 0)}")
        print(f"   - Feature columns: {status.get('feature_columns', 0)}")
        print(f"   - Background ML running: {status.get('background_ml_running', False)}")
        
    except Exception as e:
        print(f"❌ System status failed: {e}")
    
    print("\n" + "="*60 + "\n")

def test_feature_data_quality():
    """Test data quality and ranges of new features"""
    print("📊 === FEATURE DATA QUALITY TEST ===")
    
    engineer = EthereumFeatureEngineer()
    
    # Create multiple test scenarios
    scenarios = [
        {
            'name': 'Low Congestion',
            'baseFeePerGas': 15000000000,  # 15 gwei
            'gasUsed': 15000000,
            'gasLimit': 30000000,
            'network_utilization': 50.0,
            'mempool_pending_count': 80000,
        },
        {
            'name': 'High Congestion', 
            'baseFeePerGas': 60000000000,  # 60 gwei
            'gasUsed': 29000000,
            'gasLimit': 30000000,
            'network_utilization': 96.7,
            'mempool_pending_count': 250000,
        },
        {
            'name': 'Medium Congestion',
            'baseFeePerGas': 30000000000,  # 30 gwei
            'gasUsed': 22500000,
            'gasLimit': 30000000,
            'network_utilization': 75.0,
            'mempool_pending_count': 150000,
        }
    ]
    
    print("Testing feature ranges across different network conditions:\n")
    
    for scenario in scenarios:
        print(f"📋 {scenario['name']} Scenario:")
        print(f"   Network Util: {scenario['network_utilization']:.1f}%")
        print(f"   Base Fee: {scenario['baseFeePerGas']/1e9:.1f} gwei")
        print(f"   Mempool Size: {scenario['mempool_pending_count']:,}")
        
        # Add required fields
        scenario.update({
            'blockNumber': 18500000,
            'timestamp': datetime.now(),
            'mempool_total_size': scenario['mempool_pending_count'] * 500,
            'median_priority_fee': 2.0
        })
        
        df = pd.DataFrame([scenario])
        
        try:
            # Test each new feature category
            df_health = engineer.add_network_health_features(df.copy())
            df_miner = engineer.add_miner_validator_features(df.copy())
            df_tx = engineer.add_transaction_type_features(df.copy())
            
            # Network Health ranges
            print(f"   Network Health:")
            print(f"     Uncle rate: {df_health['uncle_block_rate'].iloc[0]:.4f}")
            print(f"     Validator participation: {df_health['validator_participation'].iloc[0]:.3f}")
            print(f"     Finalization delay: {df_health['finalization_delay'].iloc[0]:.1f}s")
            
            # Miner/Validator ranges
            print(f"   Miner/Validator:")
            print(f"     MEV bundle ratio: {df_miner['flashbots_bundle_ratio'].iloc[0]:.3f}")
            print(f"     Fee revenue ratio: {df_miner['fee_revenue_ratio'].iloc[0]:.3f}")
            print(f"     Private mempool ratio: {df_miner['private_mempool_ratio'].iloc[0]:.3f}")
            
            # Transaction Type ranges
            print(f"   Transaction Types:")
            print(f"     Simple transfers: {df_tx['simple_transfer_ratio'].iloc[0]:.3f}")
            print(f"     Complex contracts: {df_tx['complex_contract_ratio'].iloc[0]:.3f}")
            print(f"     Failed TX ratio: {df_tx['failed_transaction_ratio'].iloc[0]:.3f}")
            print(f"     Avg gas per TX: {df_tx['average_tx_gas_used'].iloc[0]:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("="*60 + "\n")

def test_performance_new_features():
    """Test performance impact of new features"""
    print("⚡ === PERFORMANCE TEST - NEW FEATURES ===")
    
    engineer = EthereumFeatureEngineer()
    collector = EthereumDataCollector()
    
    # Get real network data
    try:
        network_state = collector.get_current_network_state()
        print(f"✅ Using real network data from block {network_state['blockNumber']}")
    except:
        network_state = {
            'baseFeePerGas': 25000000000,
            'gasUsed': 25000000,
            'gasLimit': 30000000,
            'network_utilization': 83.3,
            'blockNumber': 18500000,
            'timestamp': datetime.now(),
            'mempool_pending_count': 150000,
            'mempool_total_size': 50000000,
            'median_priority_fee': 2.5
        }
        print("⚠️ Using mock data for performance test")
    
    df = pd.DataFrame([network_state])
    
    # Test individual feature categories
    feature_tests = [
        ('Network Health', engineer.add_network_health_features),
        ('Miner/Validator', engineer.add_miner_validator_features),
        ('Transaction Type', engineer.add_transaction_type_features),
        ('Complete Features', engineer.create_all_features)
    ]
    
    print("\nPerformance Results:")
    for name, method in feature_tests:
        times = []
        for i in range(5):  # 5 runs for average
            start_time = time.time()
            try:
                result_df = method(df.copy())
                exec_time = (time.time() - start_time) * 1000
                times.append(exec_time)
            except Exception as e:
                print(f"   ❌ {name}: Failed - {e}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"   ✅ {name}: {avg_time:.2f}ms avg ({min_time:.2f}-{max_time:.2f}ms range)")
            
            if name == 'Complete Features':
                feature_count = len(result_df.columns) - len(df.columns)
                print(f"      Features created: {feature_count}")
                print(f"      Time per feature: {avg_time/feature_count:.3f}ms")
    
    print("\n" + "="*60 + "\n")

def run_all_new_feature_tests():
    """Run all tests for newly added features"""
    print("🚀 COMPREHENSIVE TEST - NEWLY ADDED FEATURES")
    print(f"Started at: {datetime.now()}")
    print("=" * 80)
    print()
    
    try:
        test_enhanced_data_collection()
        test_new_feature_engineering()
        test_pipeline_integration()
        test_feature_data_quality()
        test_performance_new_features()
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
    print(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    run_all_new_feature_tests() 