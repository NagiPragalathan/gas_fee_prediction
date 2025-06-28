"""Ultra-fast rule-based gas fee calculations (~0.05ms)"""

from typing import Dict
from .models import GasFeeRecommendation, PriorityFeeRecommendation, SlippageRecommendation
from .config import Config, NetworkConfig

class FastRuleBasedRecommendations:
    """Ultra-fast rule-based gas fee calculations using simple heuristics"""
    
    def __init__(self):
        self.config = Config()
        self.network_config = NetworkConfig()
    
    def get_gas_fee_fast(self, base_fee: float, network_util: float, mempool_size: int) -> Dict[str, float]:
        """
        Ultra-fast rule-based gas fee calculation (~0.05ms)
        
        Args:
            base_fee: Current base fee in gwei
            network_util: Network utilization percentage
            mempool_size: Number of pending transactions
            
        Returns:
            Dictionary with gas fee recommendations for different speeds
        """
        
        # Base multiplier based on network utilization
        if network_util > self.network_config.HIGH_CONGESTION_THRESHOLD:
            multiplier = 1.5
        elif network_util > self.network_config.MEDIUM_CONGESTION_THRESHOLD:
            multiplier = 1.2
        elif network_util > self.network_config.LOW_CONGESTION_THRESHOLD:
            multiplier = 1.1
        else:
            multiplier = 1.0
            
        # Mempool pressure adjustment
        if mempool_size > self.network_config.HIGH_MEMPOOL_SIZE:
            multiplier *= 1.3
        elif mempool_size > self.network_config.MEDIUM_MEMPOOL_SIZE:
            multiplier *= 1.15
            
        return {
            'slow': base_fee * 0.9 * multiplier,
            'standard': base_fee * 1.1 * multiplier,
            'fast': base_fee * 1.4 * multiplier,
            'rapid': base_fee * 1.8 * multiplier
        }
    
    def get_priority_fee_fast(self, mempool_congestion: float, user_urgency: float) -> Dict[str, float]:
        """
        Simple heuristic-based priority fee calculation
        
        Args:
            mempool_congestion: Mempool congestion level (0-100)
            user_urgency: User urgency level (0-1)
            
        Returns:
            Dictionary with priority fee recommendations
        """
        base_priority = self.config.DEFAULT_BASE_PRIORITY_FEE
        
        congestion_multiplier = 1 + (mempool_congestion / 100)
        urgency_multiplier = 1 + (user_urgency * 0.5)
        
        final_priority = base_priority * congestion_multiplier * urgency_multiplier
        
        return {
            'low': final_priority * 0.7,
            'medium': final_priority,
            'high': final_priority * 1.8,
            'urgent': final_priority * 2.5
        }
    
    def get_slippage_fast(self, trade_size_usd: float, pool_liquidity_usd: float, 
                         volatility_score: float) -> Dict[str, float]:
        """
        Mathematical slippage calculation (no ML needed)
        
        Args:
            trade_size_usd: Trade size in USD
            pool_liquidity_usd: Available pool liquidity in USD
            volatility_score: Market volatility score (0-1)
            
        Returns:
            Dictionary with slippage tolerance recommendations
        """
        if pool_liquidity_usd <= 0:
            return {'conservative': 5.0, 'balanced': 3.0, 'aggressive': 1.5}
            
        trade_ratio = trade_size_usd / pool_liquidity_usd
        
        # Base slippage calculation
        base_slippage = trade_ratio * 100  # Convert to percentage
        
        # Volatility adjustment
        volatility_adjustment = volatility_score * 0.5
        
        # MEV risk (simple heuristic)
        mev_risk = min(trade_size_usd / 10000, 1.0) * 0.3
        
        total_slippage = base_slippage + volatility_adjustment + mev_risk
        
        return {
            'conservative': max(total_slippage * 2.0, 0.1),
            'balanced': max(total_slippage * 1.5, 0.05),
            'aggressive': max(total_slippage * 1.1, 0.01)
        }
    
    def get_complete_recommendation(self, base_fee: float, network_util: float, 
                                  mempool_size: int, trade_size_usd: float,
                                  pool_liquidity_usd: float, volatility_score: float,
                                  user_urgency: float) -> Dict:
        """
        Get complete rule-based recommendation in one call
        
        Returns:
            Complete recommendation dictionary
        """
        gas_fees = self.get_gas_fee_fast(base_fee, network_util, mempool_size)
        priority_fees = self.get_priority_fee_fast(network_util, user_urgency)
        slippage = self.get_slippage_fast(trade_size_usd, pool_liquidity_usd, volatility_score)
        
        return {
            'gas_fees': gas_fees,
            'priority_fees': priority_fees,
            'slippage': slippage,
            'source': 'rule_based'
        } 