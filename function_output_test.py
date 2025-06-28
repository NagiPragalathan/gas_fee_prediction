"""
🔍 INDIVIDUAL FUNCTION OUTPUT TEST
Shows the exact output of each data collector function
"""

import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_collector import EthereumDataCollector

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_function_output(func_name: str, output: any, execution_time: float = None):
    """Print function output in formatted way"""
    print(f"\n📋 FUNCTION: {func_name}")
    print("-" * 50)
    
    if execution_time:
        print(f"⏱️  Execution Time: {execution_time:.3f}s")
    
    if isinstance(output, dict):
        print(f"📊 Output Type: Dictionary ({len(output)} keys)")
        for key, value in output.items():
            if isinstance(value, (int, float)):
                if key.endswith('PerGas') or key.endswith('Fee'):
                    print(f"   {key}: {value:,} wei ({value/1e9:.3f} gwei)")
                elif key.endswith('utilization'):
                    print(f"   {key}: {value:.2f}%")
                elif key.endswith('count') or key.endswith('size'):
                    print(f"   {key}: {value:,}")
                else:
                    print(f"   {key}: {value}")
            elif isinstance(value, str):
                print(f"   {key}: {value[:50]}{'...' if len(str(value)) > 50 else ''}")
            elif isinstance(value, datetime):
                print(f"   {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
            elif isinstance(value, dict):
                print(f"   {key}: {{{len(value)} items}}")
            elif isinstance(value, list):
                print(f"   {key}: [{len(value)} items]")
            else:
                print(f"   {key}: {type(value).__name__}")
    elif isinstance(output, list):
        print(f"📊 Output Type: List ({len(output)} items)")
        if output:
            print(f"   First item type: {type(output[0]).__name__}")
            if len(output) > 0:
                print(f"   Sample: {str(output[0])[:100]}...")
    elif isinstance(output, (int, float)):
        print(f"📊 Output Type: {type(output).__name__}")
        print(f"   Value: {output}")
    else:
        print(f"📊 Output Type: {type(output).__name__}")
        print(f"   Value: {str(output)[:200]}...")
    
    print("-" * 50)

def test_all_functions():
    """Test each function individually and show outputs"""
    
    print("🚀 ETHEREUM DATA COLLECTOR - FUNCTION OUTPUT TEST")
    print(f"Started at: {datetime.now()}")
    
    # Initialize collector
    collector = EthereumDataCollector()
    
    # =================================================================
    print_section("CONNECTIVITY & BASIC TESTS")
    # =================================================================
    
    # Test 1: Connectivity
    try:
        start_time = time.time()
        output = collector.test_connectivity()
        exec_time = time.time() - start_time
        print_function_output("test_connectivity()", output, exec_time)
    except Exception as e:
        print_function_output("test_connectivity()", f"❌ ERROR: {e}")
    
    # Test 2: Web3 Connection Check
    try:
        start_time = time.time()
        is_connected = collector.web3.is_connected()
        current_block = collector.web3.eth.block_number if is_connected else None
        output = {
            'is_connected': is_connected,
            'current_block': current_block,
            'node_url': collector.eth_node_url
        }
        exec_time = time.time() - start_time
        print_function_output("web3.is_connected() + block_number", output, exec_time)
    except Exception as e:
        print_function_output("web3.is_connected()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("CORE NETWORK DATA COLLECTION")
    # =================================================================
    
    # Test 3: Current Network State (Full)
    try:
        start_time = time.time()
        output = collector.get_current_network_state(fast_mode=False)
        exec_time = time.time() - start_time
        print_function_output("get_current_network_state(fast_mode=False)", output, exec_time)
    except Exception as e:
        print_function_output("get_current_network_state(full)", f"❌ ERROR: {e}")
    
    # Test 4: Current Network State (Fast)
    try:
        start_time = time.time()
        output = collector.get_current_network_state(fast_mode=True)
        exec_time = time.time() - start_time
        print_function_output("get_current_network_state(fast_mode=True)", output, exec_time)
    except Exception as e:
        print_function_output("get_current_network_state(fast)", f"❌ ERROR: {e}")
    
    # Test 5: Enhanced Network State
    try:
        start_time = time.time()
        output = collector.get_enhanced_network_state()
        exec_time = time.time() - start_time
        print_function_output("get_enhanced_network_state()", output, exec_time)
    except Exception as e:
        print_function_output("get_enhanced_network_state()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("MEMPOOL & EXTERNAL APIs")
    # =================================================================
    
    # Test 6: Mempool Data
    try:
        start_time = time.time()
        output = collector._get_real_mempool_data()
        exec_time = time.time() - start_time
        print_function_output("_get_real_mempool_data()", output, exec_time)
    except Exception as e:
        print_function_output("_get_real_mempool_data()", f"❌ ERROR: {e}")
    
    # Test 7: External Gas Estimates
    try:
        start_time = time.time()
        output = collector.get_external_gas_estimates()
        exec_time = time.time() - start_time
        print_function_output("get_external_gas_estimates()", output, exec_time)
    except Exception as e:
        print_function_output("get_external_gas_estimates()", f"❌ ERROR: {e}")
    
    # Test 8: Quick Mempool Estimate
    try:
        start_time = time.time()
        output = collector._get_quick_mempool_estimate()
        exec_time = time.time() - start_time
        print_function_output("_get_quick_mempool_estimate()", output, exec_time)
    except Exception as e:
        print_function_output("_get_quick_mempool_estimate()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("PRIORITY FEE & TRANSACTION ANALYSIS")
    # =================================================================
    
    # Test 9: Priority Fee from Block
    try:
        start_time = time.time()
        latest_block = collector.web3.eth.get_block('latest', full_transactions=True)
        priority_fee = collector._calculate_median_priority_from_hashes(latest_block)
        output = {
            'block_number': latest_block.number,
            'transaction_count': len(latest_block.transactions),
            'base_fee_gwei': latest_block.baseFeePerGas / 1e9,
            'median_priority_fee_gwei': priority_fee
        }
        exec_time = time.time() - start_time
        print_function_output("_calculate_median_priority_from_hashes()", output, exec_time)
    except Exception as e:
        print_function_output("_calculate_median_priority_from_hashes()", f"❌ ERROR: {e}")
    
    # Test 10: Blocknative Priority Fee Fallback
    try:
        start_time = time.time()
        output = collector._get_blocknative_priority_fee()
        exec_time = time.time() - start_time
        print_function_output("_get_blocknative_priority_fee()", output, exec_time)
    except Exception as e:
        print_function_output("_get_blocknative_priority_fee()", f"❌ ERROR: {e}")
    
    # Test 11: Cached Priority Fee
    try:
        start_time = time.time()
        output = collector._get_cached_priority_fee()
        exec_time = time.time() - start_time
        print_function_output("_get_cached_priority_fee()", output, exec_time)
    except Exception as e:
        print_function_output("_get_cached_priority_fee()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("HISTORICAL DATA")
    # =================================================================
    
    # Test 12: Historical Data (Small Sample)
    try:
        start_time = time.time()
        output = collector.get_historical_data(hours_back=1)
        exec_time = time.time() - start_time
        
        summary = {
            'total_points': len(output),
            'time_range': f"{output[0]['timestamp']} to {output[-1]['timestamp']}" if output else "No data",
            'avg_base_fee_gwei': sum(d['baseFeePerGas'] for d in output) / (len(output) * 1e9) if output else 0,
            'avg_network_util': sum((d['gasUsed']/d['gasLimit'])*100 for d in output) / len(output) if output else 0
        }
        print_function_output("get_historical_data(hours_back=1)", summary, exec_time)
    except Exception as e:
        print_function_output("get_historical_data()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("AUTOMATION FEATURES")
    # =================================================================
    
    # Test 13: Auto Pool Liquidity
    try:
        start_time = time.time()
        output = collector._auto_get_pool_liquidity(5000, None)
        exec_time = time.time() - start_time
        print_function_output("_auto_get_pool_liquidity(5000, None)", output, exec_time)
    except Exception as e:
        print_function_output("_auto_get_pool_liquidity()", f"❌ ERROR: {e}")
    
    # Test 14: Auto Volatility Score
    try:
        start_time = time.time()
        output = collector._auto_get_volatility_score(None)
        exec_time = time.time() - start_time
        print_function_output("_auto_get_volatility_score(None)", output, exec_time)
    except Exception as e:
        print_function_output("_auto_get_volatility_score()", f"❌ ERROR: {e}")
    
    # Test 15: Auto Determine Urgency
    try:
        start_time = time.time()
        network_data = collector.get_current_network_state(fast_mode=True)
        urgency = collector._auto_determine_urgency(network_data, 5000)
        output = {
            'trade_size_usd': 5000,
            'network_utilization': network_data.get('network_utilization', 0),
            'mempool_size': network_data.get('mempool_pending_count', 0),
            'calculated_urgency': urgency
        }
        exec_time = time.time() - start_time
        print_function_output("_auto_determine_urgency(network_data, 5000)", output, exec_time)
    except Exception as e:
        print_function_output("_auto_determine_urgency()", f"❌ ERROR: {e}")
    
    # Test 16: Fully Automated Parameters
    try:
        start_time = time.time()
        output = collector.get_fully_automated_params(5000, None)
        exec_time = time.time() - start_time
        print_function_output("get_fully_automated_params(5000, None)", output, exec_time)
    except Exception as e:
        print_function_output("get_fully_automated_params()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("DEX & MARKET DATA APIs")
    # =================================================================
    
    # Test 17: Uniswap Liquidity
    try:
        start_time = time.time()
        output = collector._fetch_uniswap_liquidity('0xA0b86a33E6441786C79c8b23c5C05d5e9c2f5d3b')
        exec_time = time.time() - start_time
        print_function_output("_fetch_uniswap_liquidity(WETH)", output, exec_time)
    except Exception as e:
        print_function_output("_fetch_uniswap_liquidity()", f"❌ ERROR: {e}")
    
    # Test 18: 1inch Liquidity Estimate
    try:
        start_time = time.time()
        output = collector._fetch_1inch_liquidity_estimate()
        exec_time = time.time() - start_time
        print_function_output("_fetch_1inch_liquidity_estimate()", output, exec_time)
    except Exception as e:
        print_function_output("_fetch_1inch_liquidity_estimate()", f"❌ ERROR: {e}")
    
    # Test 19: CoinGecko ETH Volatility
    try:
        start_time = time.time()
        output = collector._fetch_eth_volatility()
        exec_time = time.time() - start_time
        print_function_output("_fetch_eth_volatility()", output, exec_time)
    except Exception as e:
        print_function_output("_fetch_eth_volatility()", f"❌ ERROR: {e}")
    
    # Test 20: Smart Liquidity Defaults
    try:
        start_time = time.time()
        outputs = {}
        for trade_size in [1000, 10000, 100000]:
            outputs[f'trade_{trade_size}'] = collector._smart_liquidity_default(trade_size)
        exec_time = time.time() - start_time
        print_function_output("_smart_liquidity_default(various sizes)", outputs, exec_time)
    except Exception as e:
        print_function_output("_smart_liquidity_default()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("NETWORK ANALYSIS FUNCTIONS")
    # =================================================================
    
    # Test 21: Mempool Composition Analysis
    try:
        start_time = time.time()
        output = collector._analyze_mempool_composition()
        exec_time = time.time() - start_time
        print_function_output("_analyze_mempool_composition()", output, exec_time)
    except Exception as e:
        print_function_output("_analyze_mempool_composition()", f"❌ ERROR: {e}")
    
    # Test 22: Block Timing Stats
    try:
        start_time = time.time()
        output = collector._get_block_timing_stats()
        exec_time = time.time() - start_time
        print_function_output("_get_block_timing_stats()", output, exec_time)
    except Exception as e:
        print_function_output("_get_block_timing_stats()", f"❌ ERROR: {e}")
    
    # Test 23: Transaction Type Estimates
    try:
        start_time = time.time()
        output = collector._estimate_transaction_types()
        exec_time = time.time() - start_time
        print_function_output("_estimate_transaction_types()", output, exec_time)
    except Exception as e:
        print_function_output("_estimate_transaction_types()", f"❌ ERROR: {e}")
    
    # Test 24: Network Volatility Estimation
    try:
        start_time = time.time()
        output = collector._estimate_volatility_from_network()
        exec_time = time.time() - start_time
        print_function_output("_estimate_volatility_from_network()", output, exec_time)
    except Exception as e:
        print_function_output("_estimate_volatility_from_network()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("OPTIMIZED PERFORMANCE FUNCTIONS")
    # =================================================================
    
    # Test 25: Optimized Network State
    try:
        start_time = time.time()
        output = collector.get_current_network_state_optimized()
        exec_time = time.time() - start_time
        print_function_output("get_current_network_state_optimized()", output, exec_time)
    except Exception as e:
        print_function_output("get_current_network_state_optimized()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("CACHE & UTILITY FUNCTIONS")
    # =================================================================
    
    # Test 26: Liquidity Cache
    try:
        start_time = time.time()
        collector._cache_liquidity("test_key", 1000000)
        cache_status = {
            'cache_size': len(collector.liquidity_cache),
            'test_key_exists': 'test_key' in collector.liquidity_cache,
            'test_value': collector.liquidity_cache.get('test_key', {}).get('value', 'N/A')
        }
        exec_time = time.time() - start_time
        print_function_output("_cache_liquidity() & cache status", cache_status, exec_time)
    except Exception as e:
        print_function_output("_cache_liquidity()", f"❌ ERROR: {e}")
    
    # Test 27: Volatility Cache
    try:
        start_time = time.time()
        collector._cache_volatility("test_vol", 0.5)
        cache_status = {
            'cache_size': len(collector.volatility_cache),
            'test_key_exists': 'test_vol' in collector.volatility_cache,
            'test_value': collector.volatility_cache.get('test_vol', {}).get('value', 'N/A')
        }
        exec_time = time.time() - start_time
        print_function_output("_cache_volatility() & cache status", cache_status, exec_time)
    except Exception as e:
        print_function_output("_cache_volatility()", f"❌ ERROR: {e}")
    
    # Test 28: Fallback Data
    try:
        start_time = time.time()
        output = collector._get_fallback_data()
        exec_time = time.time() - start_time
        print_function_output("_get_fallback_data()", output, exec_time)
    except Exception as e:
        print_function_output("_get_fallback_data()", f"❌ ERROR: {e}")
    
    # Test 29: Rate Limit Protection
    try:
        start_time = time.time()
        collector._rate_limit_protection()
        rate_limit_status = {
            'request_count': collector.request_count,
            'last_request_time': collector.last_request_time,
            'min_interval': collector.min_request_interval
        }
        exec_time = time.time() - start_time
        print_function_output("_rate_limit_protection()", rate_limit_status, exec_time)
    except Exception as e:
        print_function_output("_rate_limit_protection()", f"❌ ERROR: {e}")
    
    # =================================================================
    print_section("FINAL SUMMARY")
    # =================================================================
    
    print(f"\n🏁 FUNCTION OUTPUT TEST COMPLETED")
    print(f"⏱️  Total test time: {time.time() - start_time:.2f}s")
    print(f"📅 Completed at: {datetime.now()}")
    print("\n" + "="*80)

if __name__ == "__main__":
    start_time = time.time()
    test_all_functions() 