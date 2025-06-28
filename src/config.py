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
    
    # Model Training Settings
    HISTORICAL_DATA_HOURS = 168  # 7 days
    CACHE_VALIDITY_SECONDS = 30
    BACKGROUND_UPDATE_INTERVAL = 10  # seconds
    
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
    
    # API Keys
    ETHERSCAN_API_KEY = "P35Q6ZDPJU3FQR3NRP9BPZCFA5V6V1YMRR"
    
    # Add backup nodes
    BACKUP_NODES = [
        "https://ethereum.publicnode.com",
        "https://rpc.ankr.com/eth", 
        "https://eth.llamarpc.com",
        "https://ethereum.blockpi.network/v1/rpc/public",
        "https://cloudflare-eth.com"
    ]

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