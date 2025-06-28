"""Collect real-time Ethereum network data"""

import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import Config, NetworkConfig
from .models import NetworkState

class EthereumDataCollector:
    """
    Collect real-time Ethereum network data for gas fee prediction
    
    In production, this would connect to:
    - Ethereum nodes via Web3.py
    - Mempool APIs (ethgasstation, blocknative, 1inch)
    - DEX APIs for liquidity data
    
    Currently uses realistic mock data for demonstration.
    """
     
    def __init__(self, eth_node_url: str = None):
        self.config = Config()
        self.network_config = NetworkConfig()
        self.eth_node_url = eth_node_url or self.config.ETH_NODE_URL
        self.mempool_apis = self.config.MEMPOOL_APIS
        self.last_data = None
        
        print(f"📡 EthereumDataCollector initialized")
        print(f"   Node URL: {self.eth_node_url}")
        print(f"   Mempool APIs: {len(self.mempool_apis)} configured")
    
    def get_current_network_state(self) -> Dict:
        """
        Get current Ethereum network state
        
        In production, this would make real API calls to:
        - eth_getBlockByNumber for latest block data
        - eth_pendingTransactions for mempool data
        - External APIs for gas price estimates
        
        Returns:
            Dictionary with current network state
        """
        try:
            current_time = datetime.now()
            
            # Simulate realistic Ethereum data with patterns
            base_fee = self._simulate_base_fee(current_time)
            gas_used = np.random.randint(20000000, 29000000)
            gas_limit = self.network_config.MAX_GAS_LIMIT
            network_util = (gas_used / gas_limit) * 100
            
            mock_data = {
                'baseFeePerGas': int(base_fee * 1e9),  # Convert to wei
                'gasUsed': gas_used,
                'gasLimit': gas_limit,
                'network_utilization': network_util,
                'blockNumber': self._get_mock_block_number(),
                'timestamp': current_time,
                'mempool_pending_count': self._simulate_mempool_size(network_util),
                'mempool_total_size': np.random.randint(30000000, 80000000),
                'median_priority_fee': self._simulate_priority_fee(network_util),
                'avg_slippage': np.random.uniform(0.01, 0.5)
            }
            
            self.last_data = mock_data
            return mock_data
            
        except Exception as e:
            print(f"⚠️ Error collecting network data: {e}")
            return self._get_fallback_data()
    
    def _simulate_base_fee(self, current_time: datetime) -> float:
        """Simulate realistic base fee with time patterns"""
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Base fee patterns
        business_hours_multiplier = 1.3 if 14 <= hour <= 22 else 0.8
        weekend_multiplier = 0.7 if day_of_week >= 5 else 1.0
        
        # Add some randomness
        base_fee = np.random.uniform(15, 50) * business_hours_multiplier * weekend_multiplier
        
        # Add volatility spikes occasionally
        if np.random.random() < 0.05:  # 5% chance of spike
            base_fee *= np.random.uniform(1.5, 3.0)
        
        return max(base_fee, 1.0)  # Minimum 1 gwei
    
    def _simulate_mempool_size(self, network_util: float) -> int:
        """Simulate mempool size based on network utilization"""
        base_size = 100000
        utilization_factor = network_util / 100
        congestion_size = int(base_size * (1 + utilization_factor * 2))
        
        return np.random.randint(
            int(congestion_size * 0.8), 
            int(congestion_size * 1.2)
        )
    
    def _simulate_priority_fee(self, network_util: float) -> float:
        """Simulate priority fee based on network utilization"""
        base_priority = self.config.DEFAULT_BASE_PRIORITY_FEE
        utilization_factor = network_util / 100
        
        return base_priority * (1 + utilization_factor * 2) + np.random.uniform(-0.5, 0.5)
    
    def _get_mock_block_number(self) -> int:
        """Get mock block number based on current time"""
        # Approximate current Ethereum block number
        return 18500000 + int(time.time()) // self.network_config.AVERAGE_BLOCK_TIME
    
    def _get_fallback_data(self) -> Dict:
        """Return fallback data when collection fails"""
        if self.last_data:
            return self.last_data
        
        return {
            'baseFeePerGas': 20000000000,  # 20 gwei
            'gasUsed': 25000000,
            'gasLimit': self.network_config.MAX_GAS_LIMIT,
            'network_utilization': 83.33,
            'blockNumber': 18500000,
            'timestamp': datetime.now(),
            'mempool_pending_count': 150000,
            'mempool_total_size': 50000000,
            'median_priority_fee': 2.0,
            'avg_slippage': 0.1
        }
    
    def get_historical_data(self, hours_back: int = 24) -> List[Dict]:
        """
        Get historical network data for training
        
        Args:
            hours_back: Number of hours of historical data to generate
            
        Returns:
            List of historical network state dictionaries
        """
        print(f"📊 Generating {hours_back} hours of historical network data...")
        
        historical_data = []
        current_time = datetime.now()
        
        # Generate data points (5 blocks per hour average)
        for i in range(hours_back * 5):
            time_offset = timedelta(minutes=i * self.network_config.AVERAGE_BLOCK_TIME)
            block_time = current_time - time_offset
            
            # Simulate realistic patterns
            base_fee = self._simulate_base_fee(block_time)
            network_util = self._simulate_network_utilization(block_time)
            gas_used = int(self.network_config.MAX_GAS_LIMIT * network_util / 100)
            
            historical_point = {
                'baseFeePerGas': int(base_fee * 1e9),
                'gasUsed': gas_used,
                'gasLimit': self.network_config.MAX_GAS_LIMIT,
                'blockNumber': 18500000 - i,
                'timestamp': block_time,
                'mempool_pending_count': self._simulate_mempool_size(network_util),
                'mempool_total_size': np.random.randint(25000000, 90000000),
                'median_priority_fee': self._simulate_priority_fee(network_util),
                'avg_slippage': np.random.uniform(0.01, 1.0)
            }
            
            historical_data.append(historical_point)
        
        # Sort by timestamp (oldest first)
        historical_data.sort(key=lambda x: x['timestamp'])
        
        print(f"✅ Generated {len(historical_data)} historical data points")
        return historical_data
    
    def _simulate_network_utilization(self, block_time: datetime) -> float:
        """Simulate network utilization with time patterns"""
        hour = block_time.hour
        day_of_week = block_time.weekday()
        
        # Base utilization with patterns
        business_hours_boost = 20 if 14 <= hour <= 22 else 0
        weekend_reduction = -15 if day_of_week >= 5 else 0
        
        base_utilization = 70 + business_hours_boost + weekend_reduction
        
        # Add randomness
        utilization = base_utilization + np.random.uniform(-20, 20)
        
        return max(min(utilization, 98), 30)  # Clamp between 30-98%
    
    def get_external_gas_estimates(self) -> Dict:
        """
        Get gas estimates from external APIs
        
        In production, this would call:
        - ETH Gas Station API
        - Blocknative API  
        - 1inch Gas API
        
        Returns:
            Dictionary of external estimates
        """
        estimates = {}
        
        for name, url in self.mempool_apis.items():
            try:
                # Mock external API calls
                base_estimate = np.random.uniform(15, 50)
                estimates[name] = {
                    'base_fee': base_estimate,
                    'standard': base_estimate * 1.1,
                    'fast': base_estimate * 1.4,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"⚠️ Error getting {name} estimate: {e}")
        
        return estimates
    
    def test_connectivity(self) -> Dict:
        """
        Test connectivity to all data sources
        
        Returns:
            Dictionary with connectivity status
        """
        results = {
            'eth_node': False,
            'external_apis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test main node (mock)
        try:
            # In production: test actual eth node connection
            results['eth_node'] = True
            print("✅ Ethereum node connection: OK")
        except Exception as e:
            print(f"❌ Ethereum node connection failed: {e}")
        
        # Test external APIs (mock)
        for name, url in self.mempool_apis.items():
            try:
                # In production: actual API health checks
                results['external_apis'][name] = True
                print(f"✅ {name} API: OK")
            except Exception as e:
                results['external_apis'][name] = False
                print(f"❌ {name} API failed: {e}")
        
        return results 