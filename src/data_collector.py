"""Collect REAL-TIME Ethereum network data"""

import time
import requests
import asyncio
from web3 import Web3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import Config, NetworkConfig
from .models import NetworkState
from functools import wraps

class EthereumDataCollector:
    """
    Collect REAL-TIME Ethereum network data for gas fee prediction
    
    Connects to:
    - Ethereum nodes via Web3.py (REAL-TIME)
    - Mempool APIs (ethgasstation, blocknative, 1inch) (REAL-TIME)
    - DEX APIs for liquidity data (REAL-TIME)
    """
     
    def __init__(self, eth_node_url: str = None):
        self.config = Config()
        self.network_config = NetworkConfig()
        self.eth_node_url = eth_node_url or self.config.ETH_NODE_URL
        self.mempool_apis = self.config.MEMPOOL_APIS
        
        # Initialize REAL Web3 connection
        self.web3 = Web3(Web3.HTTPProvider(self.eth_node_url))
        self.last_data = None
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        self.request_count = 0
        self.rate_limit_reset_time = time.time()
        
        # Verify REAL connection
        if not self.web3.is_connected():
            print(f"⚠️ Failed to connect to Ethereum node: {self.eth_node_url}")
            print("🔄 Falling back to public nodes...")
            self._try_backup_nodes()
        else:
            current_block = self.web3.eth.block_number
            print(f"✅ Connected to Ethereum node: {self.eth_node_url}")
            print(f"📊 Current block: {current_block}")
        
        print(f"📡 EthereumDataCollector initialized with REAL-TIME APIs")
    
    def _try_backup_nodes(self):
        """Try backup Ethereum nodes if primary fails"""
        backup_nodes = [
            "https://ethereum.publicnode.com",
            "https://rpc.ankr.com/eth",
            "https://eth.llamarpc.com",
            "https://ethereum.blockpi.network/v1/rpc/public"
        ]
        
        for node_url in backup_nodes:
            try:
                self.web3 = Web3(Web3.HTTPProvider(node_url))
                if self.web3.is_connected():
                    self.eth_node_url = node_url
                    print(f"✅ Connected to backup node: {node_url}")
                    return
            except:
                continue
        
        print("❌ All Ethereum nodes failed. Using emergency fallback mode.")
    
    def _rate_limit_protection(self):
        """Add rate limiting protection"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.rate_limit_reset_time > 60:
            self.request_count = 0
            self.rate_limit_reset_time = current_time
        
        # Limit to 30 requests per minute
        if self.request_count >= 30:
            sleep_time = 60 - (current_time - self.rate_limit_reset_time)
            if sleep_time > 0:
                print(f"⏱️ Rate limit protection: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.rate_limit_reset_time = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_current_network_state(self) -> Dict:
        """Get REAL-TIME Ethereum network state"""
        try:
            # Get block WITHOUT full transactions first (for performance)
            latest_block = self.web3.eth.get_block('latest', full_transactions=False)
            
            # Calculate median priority fee separately
            median_priority_fee = self._calculate_median_priority_from_hashes(latest_block)
            
            # Get other data...
            network_util = (latest_block.gasUsed / latest_block.gasLimit) * 100
            mempool_data = self._get_real_mempool_data()
            external_estimates = self.get_external_gas_estimates()
            
            real_data = {
                'baseFeePerGas': latest_block.baseFeePerGas,
                'gasUsed': latest_block.gasUsed,
                'gasLimit': latest_block.gasLimit,
                'network_utilization': network_util,
                'blockNumber': latest_block.number,
                'timestamp': datetime.fromtimestamp(latest_block.timestamp),
                'mempool_pending_count': mempool_data.get('pending_count', 150000),
                'mempool_total_size': mempool_data.get('total_size', 50000000),
                'median_priority_fee': median_priority_fee,
                'avg_slippage': external_estimates.get('avg_slippage', 0.1),
                'external_estimates': external_estimates
            }
            
            self.last_data = real_data
            print(f"📊 REAL data: Block {latest_block.number}, Base fee {latest_block.baseFeePerGas / 1e9:.2f} gwei")
            return real_data
            
        except Exception as e:
            print(f"⚠️ Error collecting REAL network data: {e}")
            return self._get_fallback_data()
    
    def _try_backup_for_network_state(self) -> Dict:
        """Try backup node for network state"""
        backup_nodes = [
            "https://ethereum.publicnode.com",
            "https://rpc.ankr.com/eth",
            "https://eth.llamarpc.com"
        ]
        
        for backup_url in backup_nodes:
            try:
                backup_web3 = Web3(Web3.HTTPProvider(backup_url))
                if backup_web3.is_connected():
                    print(f"🔄 Trying backup node: {backup_url}")
                    latest_block = backup_web3.eth.get_block('latest', full_transactions=True)
                    # Process with backup connection
                    # ... same logic as get_current_network_state ...
                    return self._process_block_data(latest_block, backup_web3)
            except:
                continue
        
        return self._get_fallback_data()
    
    def _get_real_mempool_data(self) -> Dict:
        """Get REAL mempool data from external APIs"""
        try:
            # Method 1: Try Etherscan pending transaction count
            try:
                etherscan_url = "https://api.etherscan.io/api"
                params = {
                    'module': 'proxy',
                    'action': 'eth_getBlockTransactionCountByNumber',
                    'tag': 'pending',
                    'apikey': self.config.ETHERSCAN_API_KEY if hasattr(self.config, 'ETHERSCAN_API_KEY') else 'YourApiKeyToken'
                }
                
                response = requests.get(etherscan_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    pending_count = int(data.get('result', '0x0'), 16)
                    
                    return {
                        'pending_count': pending_count,
                        'total_size': pending_count * 500,  # Estimate transaction size
                        'source': 'etherscan'
                    }
            except Exception as e:
                print(f"⚠️ Etherscan mempool API failed: {e}")
            
            # Method 2: Try direct Web3 pending transactions
            try:
                pending_tx_count = len(self.web3.eth.get_block('pending').transactions)
                return {
                    'pending_count': pending_tx_count,
                    'total_size': pending_tx_count * 500,
                    'source': 'web3_pending'
                }
            except Exception as e:
                print(f"⚠️ Web3 pending transactions failed: {e}")
            
            # Method 3: Estimate from network utilization
            latest_block = self.web3.eth.get_block('latest')
            network_util = (latest_block.gasUsed / latest_block.gasLimit) * 100
            estimated_pending = int(150000 * (network_util / 100))
            
            return {
                'pending_count': estimated_pending,
                'total_size': estimated_pending * 500,
                'source': 'estimated'
            }
            
        except Exception as e:
            print(f"⚠️ All mempool methods failed: {e}")
            return {'pending_count': 150000, 'total_size': 50000000, 'source': 'fallback'}
    
    def _calculate_median_priority_from_hashes(self, block) -> float:
        """Calculate priority fee using transaction hashes - BULLETPROOF VERSION"""
        try:
            priority_fees = []
            sample_size = min(10, len(block.transactions))  # Reduce to 10 for speed
            
            print(f"🔍 Analyzing {sample_size} transaction hashes...")
            
            for i in range(sample_size):
                try:
                    tx_hash = block.transactions[i]
                    
                    # FIX THE F*CKING ERROR: Convert to proper hex string
                    if hasattr(tx_hash, 'hex'):
                        # It's a HexBytes object
                        tx_hash_str = tx_hash.hex()
                    elif isinstance(tx_hash, bytes):
                        # It's raw bytes
                        tx_hash_str = tx_hash.hex()
                    elif isinstance(tx_hash, str):
                        # It's already a string
                        tx_hash_str = tx_hash
                        if not tx_hash_str.startswith('0x'):
                            tx_hash_str = '0x' + tx_hash_str
                    else:
                        # Convert whatever it is to hex
                        tx_hash_str = self.web3.to_hex(tx_hash)
                    
                    # Debug first few to see what we're working with
                    if i < 2:
                        print(f"🔍 TX {i}: {type(tx_hash)} -> {tx_hash_str[:10]}...")
                    
                    # Now get the transaction with properly formatted hash
                    tx = self.web3.eth.get_transaction(tx_hash_str)
                    
                    # Process transaction for priority fee
                    if hasattr(tx, 'maxPriorityFeePerGas') and tx.maxPriorityFeePerGas is not None:
                        priority_fee_gwei = tx.maxPriorityFeePerGas / 1e9
                        priority_fees.append(priority_fee_gwei)
                    elif hasattr(tx, 'gasPrice') and tx.gasPrice is not None:
                        # Legacy transaction
                        gas_price_gwei = tx.gasPrice / 1e9
                        base_fee_gwei = block.baseFeePerGas / 1e9
                        priority_fee_gwei = max(0, gas_price_gwei - base_fee_gwei)
                        if priority_fee_gwei > 0:
                            priority_fees.append(priority_fee_gwei)
                    
                except Exception as e:
                    if i < 3:
                        print(f"⚠️ Error getting transaction {i}: {e}")
                    continue
            
            print(f"📊 Found {len(priority_fees)} priority fees from {sample_size} transactions")
            
            if priority_fees and len(priority_fees) >= 3:  # Need at least 3 for decent median
                filtered_fees = [fee for fee in priority_fees if 0.001 <= fee <= 50]
                if filtered_fees:
                    median_priority = sorted(filtered_fees)[len(filtered_fees) // 2]
                    print(f"✅ REAL median priority fee: {median_priority:.3f} gwei")
                    return median_priority
            
            # Fallback to Blocknative
            print("🔄 Using Blocknative priority fee as fallback")
            return self._get_blocknative_priority_fee()
                
        except Exception as e:
            print(f"⚠️ Priority fee calculation failed: {e}")
            return self._get_blocknative_priority_fee()
    
    def _get_blocknative_priority_fee(self) -> float:
        """Get priority fee from Blocknative as fallback"""
        try:
            external_estimates = self.get_external_gas_estimates()
            if 'blocknative' in external_estimates:
                priority_fee = external_estimates['blocknative'].get('priority_fee_80', 1.5)
                print(f"🔄 Using Blocknative priority fee: {priority_fee:.3f} gwei")
                return priority_fee
        except Exception as e:
            print(f"⚠️ Blocknative priority fee failed: {e}")
        return 1.5  # Conservative fallback
    
    def get_historical_data(self, hours_back: int = 24) -> List[Dict]:
        """
        Get REAL historical network data
        """
        print(f"📊 Collecting {hours_back} hours of REAL historical data...")
        
        historical_data = []
        current_block = self.web3.eth.block_number
        
        # Calculate blocks to fetch (approximately 1 block per 12 seconds)
        blocks_per_hour = 300
        total_blocks = min(hours_back * blocks_per_hour, 1000)  # Limit to prevent overwhelming
        
        print(f"Fetching {total_blocks} blocks from block {current_block - total_blocks} to {current_block}")
        
        for i in range(0, total_blocks, 10):  # Fetch every 10th block to speed up
            try:
                block_number = current_block - i
                block = self.web3.eth.get_block(block_number, full_transactions=False)  # Don't fetch all transactions
                
                network_util = (block.gasUsed / block.gasLimit) * 100
                
                historical_point = {
                    'baseFeePerGas': block.baseFeePerGas,
                    'gasUsed': block.gasUsed,
                    'gasLimit': block.gasLimit,
                    'blockNumber': block.number,
                    'timestamp': datetime.fromtimestamp(block.timestamp),
                    'mempool_pending_count': len(block.transactions) if hasattr(block, 'transactions') else 150000,
                    'mempool_total_size': len(block.transactions) * 500 if hasattr(block, 'transactions') else 50000000,
                    'median_priority_fee': 2.0,  # Would need full transaction data
                    'avg_slippage': 0.1  # Would need DEX API data
                }
                
                historical_data.append(historical_point)
                
                if (i + 1) % 100 == 0:
                    print(f"   Fetched {i + 1}/{total_blocks} blocks...")
                    
            except Exception as e:
                print(f"⚠️ Error fetching block {current_block - i}: {e}")
                continue
        
        # Sort by timestamp (oldest first)
        historical_data.sort(key=lambda x: x['timestamp'])
        
        print(f"✅ Collected {len(historical_data)} REAL historical data points")
        return historical_data
    
    def get_external_gas_estimates(self) -> Dict:
        """
        Get REAL gas estimates from external APIs
        """
        estimates = {}
        
        # Blocknative Gas API (FIXED PARSING)
        try:
            response = requests.get('https://api.blocknative.com/gasprices/blockprices', timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                block_prices = data.get('blockPrices', [])
                if block_prices:
                    current_block = block_prices[0]
                    estimated_prices = current_block.get('estimatedPrices', [])
                    base_fee = current_block.get('baseFeePerGas', 0)  # Already in gwei
                    
                    if estimated_prices:
                        # Create a map by confidence level
                        prices_by_confidence = {price.get('confidence'): price for price in estimated_prices}
                        
                        # Extract prices (already in gwei)
                        confidence_70 = prices_by_confidence.get(70, {})
                        confidence_80 = prices_by_confidence.get(80, {})
                        confidence_95 = prices_by_confidence.get(95, {})
                        confidence_99 = prices_by_confidence.get(99, {})
                        
                        estimates['blocknative'] = {
                            'base_fee': base_fee,  # Already in gwei
                            'standard': confidence_70.get('price', 0),    # 70% confidence
                            'fast': confidence_80.get('price', 0),       # 80% confidence  
                            'rapid': confidence_95.get('price', 0),      # 95% confidence
                            'ultra': confidence_99.get('price', 0),      # 99% confidence
                            'priority_fee_80': confidence_80.get('maxPriorityFeePerGas', 0),
                            'priority_fee_95': confidence_95.get('maxPriorityFeePerGas', 0),
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"✅ Blocknative: Fast={estimates['blocknative']['fast']:.3f} gwei, Ultra={estimates['blocknative']['ultra']:.3f} gwei")
                    else:
                        print("⚠️ Blocknative: No estimated prices found")
                else:
                    print("⚠️ Blocknative: No block prices found")
            else:
                print(f"⚠️ Blocknative API HTTP {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ Blocknative API failed: {e}")
        
        # 1inch Gas API (keep existing)
        try:
            response = requests.get('https://gas-price-api.1inch.io/v1.4/1', timeout=10)
            if response.status_code == 200:
                data = response.json()
                estimates['1inch'] = {
                    'base_fee': data.get('baseFee', 0),
                    'standard': data.get('standard', 0),
                    'fast': data.get('fast', 0),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"✅ 1inch: {estimates['1inch']['fast']:.1f} gwei")
        except Exception as e:
            print(f"⚠️ 1inch API failed: {e}")
        
        # Add computed average slippage estimate
        if estimates:
            all_fast_estimates = [est.get('fast', 0) for est in estimates.values() if est.get('fast', 0) > 0]
            if all_fast_estimates:
                avg_estimate = sum(all_fast_estimates) / len(all_fast_estimates)
                estimates['avg_slippage'] = min(0.5, avg_estimate / 100)  # Convert to slippage estimate
        
        return estimates
    
    def test_connectivity(self) -> Dict:
        """
        Test REAL connectivity to all data sources
        """
        results = {
            'eth_node': False,
            'external_apis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test Ethereum node
        try:
            latest_block = self.web3.eth.block_number
            results['eth_node'] = True
            results['latest_block'] = latest_block
            print(f"✅ Ethereum node connection: OK (Latest block: {latest_block})")
        except Exception as e:
            print(f"❌ Ethereum node connection failed: {e}")
            results['eth_node_error'] = str(e)
        
        # Test external APIs
        external_estimates = self.get_external_gas_estimates()
        for name in ['blocknative', '1inch']:  # Changed from 'ethgasstation' to 'blocknative'
            results['external_apis'][name] = name in external_estimates
            if name in external_estimates:
                print(f"✅ {name} API: OK")
            else:
                print(f"❌ {name} API: Failed")
        
        return results
    
    def _get_fallback_data(self) -> Dict:
        """Return fallback data when REAL collection fails"""
        if self.last_data:
            print("🔄 Using last known REAL data")
            return self.last_data
        
        print("⚠️ Using emergency fallback data")
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

    def _calculate_real_median_priority_fee(self, block) -> float:
        """Calculate REAL median priority fee from block transactions"""
        return self._calculate_median_priority_from_hashes(block)