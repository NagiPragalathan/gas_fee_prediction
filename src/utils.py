"""Utility functions for Gas Fee Prediction System"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from .models import PredictionResult, ModelPerformance

def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if isinstance(result, dict):
            result['execution_time_ms'] = execution_time
        
        return result
    return wrapper

def validate_trade_parameters(trade_params: Dict) -> Dict:
    """
    Validate and sanitize trade parameters
    
    Args:
        trade_params: Dictionary of trade parameters
        
    Returns:
        Validated and sanitized parameters
    """
    validated = {}
    
    # Base fee validation
    base_fee = trade_params.get('base_fee', 20.0)
    validated['base_fee'] = max(0.1, min(base_fee, 1000.0))  # 0.1 to 1000 gwei
    
    # Network utilization validation
    network_util = trade_params.get('network_util', 80.0)
    validated['network_util'] = max(0.0, min(network_util, 100.0))  # 0-100%
    
    # Mempool size validation
    mempool_size = trade_params.get('mempool_size', 150000)
    validated['mempool_size'] = max(0, min(mempool_size, 1000000))  # 0 to 1M transactions
    
    # Trade size validation
    trade_size_usd = trade_params.get('trade_size_usd', 1000.0)
    validated['trade_size_usd'] = max(1.0, min(trade_size_usd, 1000000000.0))  # $1 to $1B
    
    # Pool liquidity validation
    pool_liquidity_usd = trade_params.get('pool_liquidity_usd', 1000000.0)
    validated['pool_liquidity_usd'] = max(1000.0, pool_liquidity_usd)  # Minimum $1K
    
    # Volatility score validation
    volatility_score = trade_params.get('volatility_score', 0.5)
    validated['volatility_score'] = max(0.0, min(volatility_score, 1.0))  # 0-1
    
    # User urgency validation
    user_urgency = trade_params.get('user_urgency', 0.5)
    validated['user_urgency'] = max(0.0, min(user_urgency, 1.0))  # 0-1
    
    return validated

def calculate_performance_metrics(latencies: List[float]) -> ModelPerformance:
    """
    Calculate performance metrics from latency measurements
    
    Args:
        latencies: List of latency measurements in milliseconds
        
    Returns:
        ModelPerformance object with calculated metrics
    """
    if not latencies:
        return ModelPerformance(
            avg_latency_ms=999.0,
            p50_latency_ms=999.0,
            p95_latency_ms=999.0,
            p99_latency_ms=999.0,
            success_rate=0.0,
            total_requests=0
        )
    
    # Filter out failed requests (marked with 999ms)
    successful_latencies = [lat for lat in latencies if lat < 999]
    success_rate = len(successful_latencies) / len(latencies) * 100
    
    if not successful_latencies:
        successful_latencies = [999.0]
    
    return ModelPerformance(
        avg_latency_ms=np.mean(successful_latencies),
        p50_latency_ms=np.percentile(successful_latencies, 50),
        p95_latency_ms=np.percentile(successful_latencies, 95),
        p99_latency_ms=np.percentile(successful_latencies, 99),
        success_rate=success_rate,
        total_requests=len(latencies)
    )

def format_gas_recommendation(recommendation: Dict) -> str:
    """
    Format gas recommendation for display
    
    Args:
        recommendation: Recommendation dictionary
        
    Returns:
        Formatted string representation
    """
    output = []
    
    if 'gas_fees' in recommendation:
        output.append("⛽ Gas Fees (gwei):")
        for speed, fee in recommendation['gas_fees'].items():
            output.append(f"   {speed.capitalize()}: {fee:.2f}")
    
    if 'priority_fees' in recommendation:
        output.append("🎯 Priority Fees (gwei):")
        for priority, fee in recommendation['priority_fees'].items():
            output.append(f"   {priority.capitalize()}: {fee:.2f}")
    
    if 'slippage' in recommendation:
        output.append("📈 Slippage Tolerance (%):")
        for tolerance, slippage in recommendation['slippage'].items():
            output.append(f"   {tolerance.capitalize()}: {slippage:.2f}%")
    
    # Add metadata
    output.append(f"🔧 Source: {recommendation.get('source', 'unknown')}")
    output.append(f"🎯 Confidence: {recommendation.get('confidence', 'unknown')}")
    output.append(f"⚡ Latency: {recommendation.get('latency_ms', 0):.2f}ms")
    
    return "\n".join(output)

def wei_to_gwei(wei_value: int) -> float:
    """Convert wei to gwei"""
    return wei_value / 1e9

def gwei_to_wei(gwei_value: float) -> int:
    """Convert gwei to wei"""
    return int(gwei_value * 1e9)

def calculate_transaction_cost(gas_limit: int, gas_price_gwei: float, priority_fee_gwei: float = 0) -> Dict:
    """
    Calculate total transaction cost
    
    Args:
        gas_limit: Gas limit for transaction
        gas_price_gwei: Gas price in gwei
        priority_fee_gwei: Priority fee in gwei
        
    Returns:
        Dictionary with cost breakdown
    """
    total_gas_price = gas_price_gwei + priority_fee_gwei
    cost_wei = gas_limit * gwei_to_wei(total_gas_price)
    cost_eth = cost_wei / 1e18
    
    return {
        'gas_limit': gas_limit,
        'gas_price_gwei': gas_price_gwei,
        'priority_fee_gwei': priority_fee_gwei,
        'total_gas_price_gwei': total_gas_price,
        'cost_wei': cost_wei,
        'cost_eth': cost_eth,
        'cost_usd': cost_eth * 2000  # Assuming $2000 ETH price
    }

def create_sample_trade_scenarios() -> List[Dict]:
    """Create sample trading scenarios for testing"""
    return [
        {
            'name': 'Small DeFi Swap',
            'params': {
                'base_fee': 20.0,
                'network_util': 70.0,
                'mempool_size': 120000,
                'trade_size_usd': 500,
                'pool_liquidity_usd': 500000,
                'volatility_score': 0.3,
                'user_urgency': 0.3
            }
        },
        {
            'name': 'Large Arbitrage Trade',
            'params': {
                'base_fee': 45.0,
                'network_util': 95.0,
                'mempool_size': 250000,
                'trade_size_usd': 50000,
                'pool_liquidity_usd': 2000000,
                'volatility_score': 0.8,
                'user_urgency': 0.9
            }
        },
        {
            'name': 'NFT Purchase',
            'params': {
                'base_fee': 30.0,
                'network_util': 85.0,
                'mempool_size': 180000,
                'trade_size_usd': 2000,
                'pool_liquidity_usd': 1000000,
                'volatility_score': 0.6,
                'user_urgency': 0.7
            }
        },
        {
            'name': 'Low Activity Period',
            'params': {
                'base_fee': 12.0,
                'network_util': 45.0,
                'mempool_size': 80000,
                'trade_size_usd': 1000,
                'pool_liquidity_usd': 800000,
                'volatility_score': 0.2,
                'user_urgency': 0.2
            }
        },
        {
            'name': 'Network Congestion',
            'params': {
                'base_fee': 80.0,
                'network_util': 98.0,
                'mempool_size': 300000,
                'trade_size_usd': 10000,
                'pool_liquidity_usd': 1500000,
                'volatility_score': 0.9,
                'user_urgency': 0.8
            }
        }
    ]

def export_performance_report(performance_stats: Dict, filename: str = None) -> str:
    """
    Export performance statistics to a report
    
    Args:
        performance_stats: Performance statistics dictionary
        filename: Optional filename for export
        
    Returns:
        Report as string
    """
    if filename is None:
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    report = []
    report.append("GAS FEE PREDICTION SYSTEM - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for method, stats in performance_stats.items():
        report.append(f"🔹 {method.upper()}:")
        report.append(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")
        report.append(f"   Avg Latency:  {stats.get('avg_latency_ms', 0):.3f}ms")
        report.append(f"   P50 Latency:  {stats.get('p50_latency_ms', 0):.3f}ms")
        report.append(f"   P95 Latency:  {stats.get('p95_latency_ms', 0):.3f}ms")
        report.append(f"   P99 Latency:  {stats.get('p99_latency_ms', 0):.3f}ms")
        report.append(f"   Total Requests: {stats.get('total_requests', 0)}")
        report.append("")
    
    report_text = "\n".join(report)
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            f.write(report_text)
        print(f"📄 Performance report saved to {filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    return report_text

def health_check_system(pipeline) -> Dict:
    """
    Perform system health check
    
    Args:
        pipeline: GasFeeCompletePipeline instance
        
    Returns:
        Health check results
    """
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'components': {},
        'recommendations': []
    }
    
    # Check data collector
    try:
        network_state = pipeline.data_collector.get_current_network_state()
        health_status['components']['data_collector'] = 'healthy'
    except Exception as e:
        health_status['components']['data_collector'] = f'error: {e}'
        health_status['overall_status'] = 'degraded'
    
    # Check fast rules engine
    try:
        test_params = {
            'base_fee': 25.0,
            'network_util': 80.0,
            'mempool_size': 150000,
            'trade_size_usd': 1000,
            'pool_liquidity_usd': 1000000,
            'volatility_score': 0.5,
            'user_urgency': 0.5
        }
        result = pipeline.get_instant_recommendation(test_params)
        health_status['components']['fast_rules'] = 'healthy'
    except Exception as e:
        health_status['components']['fast_rules'] = f'error: {e}'
        health_status['overall_status'] = 'degraded'
    
    # Check cached ML
    cache_stats = pipeline.cached_ml.get_cache_stats()
    if cache_stats['is_valid']:
        health_status['components']['cached_ml'] = 'healthy'
    else:
        health_status['components']['cached_ml'] = 'cache_expired'
        health_status['recommendations'].append('Cache needs refresh')
    
    # Check background ML thread
    if pipeline.is_running:
        health_status['components']['background_ml'] = 'running'
    else:
        health_status['components']['background_ml'] = 'stopped'
        health_status['recommendations'].append('Background ML not running')
    
    # Check trained models
    if pipeline.gas_fee_models:
        health_status['components']['trained_models'] = f'{len(pipeline.gas_fee_models)} models loaded'
    else:
        health_status['components']['trained_models'] = 'no_models'
        health_status['recommendations'].append('No trained models available')
    
    return health_status 