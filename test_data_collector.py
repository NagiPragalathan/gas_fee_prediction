"""
Simple test for src/data_collector.py - Shows actual API outputs
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import EthereumDataCollector

def test_connectivity():
    """Test connectivity and show results"""
    print("=== CONNECTIVITY TEST ===")
    
    collector = EthereumDataCollector()
    results = collector.test_connectivity()
    
    print(f"Results: {json.dumps(results, indent=2, default=str)}")
    print()

def test_current_network_state():
    """Test real-time network state collection"""
    print("=== CURRENT NETWORK STATE TEST ===")
    
    collector = EthereumDataCollector()
    
    start_time = time.time()
    network_state = collector.get_current_network_state()
    collection_time = (time.time() - start_time) * 1000
    
    print(f"Collection time: {collection_time:.2f}ms")
    print(f"Network state data:")
    print(json.dumps(network_state, indent=2, default=str))
    print()

def test_mempool_data():
    """Test mempool data collection"""
    print("=== MEMPOOL DATA TEST ===")
    
    collector = EthereumDataCollector()
    mempool_data = collector._get_real_mempool_data()
    
    print(f"Mempool data:")
    print(json.dumps(mempool_data, indent=2))
    print()

def test_external_gas_estimates():
    """Test external gas estimation APIs"""
    print("=== EXTERNAL GAS ESTIMATES TEST ===")
    
    collector = EthereumDataCollector()
    external_estimates = collector.get_external_gas_estimates()
    
    print(f"External estimates:")
    print(json.dumps(external_estimates, indent=2, default=str))
    print()

def test_historical_data_sample():
    """Test historical data collection (small sample)"""
    print("=== HISTORICAL DATA TEST (1 hour sample) ===")
    
    collector = EthereumDataCollector()
    
    start_time = time.time()
    historical_data = collector.get_historical_data(hours_back=1)
    collection_time = time.time() - start_time
    
    print(f"Collection time: {collection_time:.1f}s")
    print(f"Number of data points: {len(historical_data)}")
    
    if historical_data:
        print(f"First data point:")
        print(json.dumps(historical_data[0], indent=2, default=str))
        print(f"Last data point:")
        print(json.dumps(historical_data[-1], indent=2, default=str))
    print()

def test_priority_fee_calculation():
    """Test priority fee calculation from real block"""
    print("=== PRIORITY FEE CALCULATION TEST ===")
    
    collector = EthereumDataCollector()
    
    try:
        latest_block = collector.web3.eth.get_block('latest', full_transactions=True)
        print(f"Block number: {latest_block.number}")
        print(f"Number of transactions: {len(latest_block.transactions)}")
        print(f"Base fee: {latest_block.baseFeePerGas / 1e9:.3f} gwei")
        
        priority_fee = collector._calculate_real_median_priority_fee(latest_block)
        print(f"Calculated median priority fee: {priority_fee:.3f} gwei")
        
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_performance_multiple_calls():
    """Test performance with multiple calls"""
    print("=== PERFORMANCE TEST (5 calls) ===")
    
    collector = EthereumDataCollector()
    
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            network_state = collector.get_current_network_state()
            collection_time = (time.time() - start_time) * 1000
            times.append(collection_time)
            print(f"Call {i+1}: {collection_time:.2f}ms - Block {network_state['blockNumber']}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average time: {avg_time:.2f}ms")
    print()

def test_fixes():
    collector = EthereumDataCollector()
    
    print("🧪 Testing priority fee detection...")
    try:
        latest_block = collector.web3.eth.get_block('latest', full_transactions=True)
        priority_fee = collector._calculate_real_median_priority_fee(latest_block)
        print(f"Priority fee result: {priority_fee}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n🧪 Testing network state with rate limiting...")
    network_state = collector.get_current_network_state()
    print(f"Network state collected: {network_state.get('blockNumber', 'Failed')}")

def run_all_tests():
    """Run all tests and show raw outputs"""
    print("DATA COLLECTOR API OUTPUT TEST")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    print()
    
    test_connectivity()
    test_current_network_state()
    test_mempool_data()
    test_external_gas_estimates()
    test_historical_data_sample()
    test_priority_fee_calculation()
    test_performance_multiple_calls()
    
    print("=" * 60)
    print(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    # test_fixes()
    run_all_tests()