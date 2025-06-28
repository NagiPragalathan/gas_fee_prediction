"""Configuration settings for Gas Fee Prediction System"""

import os
from typing import Dict, List

class Config:
    """Configuration settings"""
    
    # Ethereum Node Settings
    ETH_NODE_URL = "https://eth-mainnet.g.alchemy.com/v2/_TKdoChuQW7COEa9NHXg7j3lPq9JDZZM"
    
    # External API URLs
    MEMPOOL_APIS = {
        'ethgasstation': 'https://ethgasstation.info/api/ethgasAPI.json',
        'blocknative': 'https://api.blocknative.com/gasprices/blockprices',
        '1inch': 'https://gas-price-api.1inch.io/v1.4/1'
    }
    
    # ✅ NEW: DEX and Market Data APIs
    DEX_APIS = {
        'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        'sushiswap': 'https://api.thegraph.com/subgraphs/name/sushiswap/exchange',
        '1inch': 'https://api.1inch.io/v5.0/1',
        'coingecko': 'https://api.coingecko.com/api/v3',
        'defipulse': 'https://data-api.defipulse.com/api/v1'
    }
    
    # Model Training Settings
    HISTORICAL_DATA_HOURS = 168  # 7 days
    CACHE_VALIDITY_SECONDS = 300  # 5 minutes
    BACKGROUND_UPDATE_INTERVAL = 10  # seconds
    
    # ✅ NEW: Automation Cache Settings
    LIQUIDITY_CACHE_DURATION = 300  # 5 minutes
    VOLATILITY_CACHE_DURATION = 600  # 10 minutes
    
    # LightGBM Parameters
    LIGHTGBM_PARAMS = {
        "max_depth": 6,
        "min_data_in_leaf": 10,
        "learning_rate": 0.01,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        'n_estimators': 500
    }
    
    # Feature Engineering
    QUANTILES = [0.1, 0.5, 0.9]  # Conservative, median, aggressive
    
    # Default Values
    DEFAULT_POOL_LIQUIDITY = 1000000  # $1M
    DEFAULT_VOLATILITY_SCORE = 0.5
    DEFAULT_BASE_PRIORITY_FEE = 2.0  # gwei
    
    # ✅ NEW: Automation Smart Defaults
    SMART_LIQUIDITY_TIERS = {
        1000: 500000,      # $1K trade -> $500K liquidity
        10000: 2000000,    # $10K trade -> $2M liquidity  
        100000: 10000000,  # $100K trade -> $10M liquidity
        1000000: 50000000  # $1M+ trade -> $50M liquidity
    }
    
    # API Keys
    ETHERSCAN_API_KEY = "P35Q6ZDPJU3FQR3NRP9BPZCFA5V6V1YMRR"
    ONEINCH_API_KEY = "uzF2lXeO9pYtpjthDs0ltrkVwDcup6bd"
    
    # Add backup nodes
    BACKUP_NODES = [
        "https://ethereum.publicnode.com",
        "https://rpc.ankr.com/eth", 
        "https://eth.llamarpc.com",
        "https://ethereum.blockpi.network/v1/rpc/public",
        "https://cloudflare-eth.com"
    ]

    # Performance settings
    FAST_MODE = True  # Enable fast mode by default
    MAX_PRIORITY_SAMPLES = 5  # Reduce from 10 to 5 for speed
    SKIP_HEAVY_ANALYSIS = True  # Skip time-consuming analysis
    CACHE_DURATION = 30  # Cache results for 30 seconds

class NetworkConfig:
    """Ethereum network specific configuration"""
    
    # Block timing
    AVERAGE_BLOCK_TIME = 12  # seconds
    BLOCKS_PER_HOUR = 300
    BLOCKS_PER_DAY = 7200
    
    # Gas settings
    MAX_GAS_LIMIT = 30000000
    BASE_GAS_TARGET = 15000000
    
    # Network utilization thresholds
    HIGH_CONGESTION_THRESHOLD = 90  # %
    MEDIUM_CONGESTION_THRESHOLD = 70  # %
    LOW_CONGESTION_THRESHOLD = 50  # %
    
    # Mempool thresholds
    HIGH_MEMPOOL_SIZE = 100000  # transactions
    MEDIUM_MEMPOOL_SIZE = 50000  # transactions

    # ✅ NEW: Urgency calculation thresholds
    URGENCY_THRESHOLDS = {
        'high_congestion': 95,      # Network utilization %
        'medium_congestion': 85,    # Network utilization %
        'large_trade': 50000,       # USD
        'medium_trade': 10000,      # USD
        'high_mempool': 200000,     # Transaction count
    } 