"""
Real-Time Gas Fee Estimation API
Production-ready FastAPI for trading applications

Endpoints:
- GET /health - Health check
- GET /network/status - Current network status  
- POST /gas/ultra-fast - Ultra-fast rule-based (~0.05ms) ✅ AUTOMATED
- POST /gas/cached-ml - Hybrid cached ML (~1-2ms) ✅ AUTOMATED
- POST /gas/full-ai - Complete AI analysis (80+ features) ✅ AUTOMATED
- POST /gas/all - All three recommendations ✅ AUTOMATED
- POST /gas/automated - Simplified automated endpoint ✅ AUTOMATED
- GET /gas/stream - WebSocket for real-time updates
- GET /cache/status - Get ML cache status for debugging
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import asyncio
import threading
import time
import sys
import os
import json
import logging
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import EthereumDataCollector
from src.feature_engineer import EthereumFeatureEngineer
from src.fast_rules import FastRuleBasedRecommendations
from src.cached_ml import CachedMLRecommendations
from src.pipeline import GasFeeCompletePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== UPDATED PYDANTIC MODELS =====

class AutomatedTradeRequest(BaseModel):
    """✅ AUTOMATED: Minimal trade parameters - everything else automated"""
    trade_size_usd: float = Field(..., description="Trade size in USD (ONLY required parameter!)")
    token_address: Optional[str] = Field(None, description="Optional token address for better estimates")
    
    # ✅ ALL OPTIONAL - Will be automated if not provided
    base_fee: Optional[float] = Field(None, description="Current base fee in gwei (auto-detected if not provided)")
    network_util: Optional[float] = Field(None, description="Network utilization % (auto-detected if not provided)")
    mempool_size: Optional[int] = Field(None, description="Mempool size (auto-detected if not provided)")
    pool_liquidity_usd: Optional[float] = Field(None, description="Pool liquidity in USD (auto-detected if not provided)")
    volatility_score: Optional[float] = Field(None, description="Volatility score 0-1 (auto-detected if not provided)")
    user_urgency: Optional[float] = Field(None, description="User urgency 0-1 (auto-determined if not provided)")

class LegacyTradeRequest(BaseModel):
    """Legacy trade parameters for backwards compatibility"""
    base_fee: Optional[float] = Field(None, description="Current base fee in gwei (auto-detected if not provided)")
    network_util: Optional[float] = Field(None, description="Network utilization % (auto-detected if not provided)")
    mempool_size: Optional[int] = Field(None, description="Mempool size (auto-detected if not provided)")
    trade_size_usd: float = Field(5000, description="Trade size in USD")
    pool_liquidity_usd: Optional[float] = Field(None, description="Pool liquidity in USD (auto-detected if not provided)")
    volatility_score: Optional[float] = Field(None, description="Volatility score (auto-detected if not provided)")
    user_urgency: Optional[float] = Field(None, description="User urgency (auto-determined if not provided)")
    token_address: Optional[str] = Field(None, description="Optional token address for better estimates")

class GasFeesResponse(BaseModel):
    """Gas fee recommendations"""
    slow: float = Field(description="Slow gas price in gwei")
    standard: float = Field(description="Standard gas price in gwei")
    fast: float = Field(description="Fast gas price in gwei")
    rapid: Optional[float] = Field(None, description="Rapid gas price in gwei")

class PriorityFeesResponse(BaseModel):
    """Priority fee recommendations"""
    low: float = Field(description="Low priority fee in gwei")
    medium: float = Field(description="Medium priority fee in gwei")
    high: float = Field(description="High priority fee in gwei")
    urgent: Optional[float] = Field(None, description="Urgent priority fee in gwei")

class SlippageResponse(BaseModel):
    """Slippage estimates"""
    conservative: float = Field(description="Conservative slippage %")
    balanced: float = Field(description="Balanced slippage %")
    aggressive: float = Field(description="Aggressive slippage %")

class NetworkStatus(BaseModel):
    """Current network status"""
    block_number: int = Field(description="Latest block number")
    base_fee_gwei: float = Field(description="Current base fee in gwei")
    network_utilization: float = Field(description="Network utilization %")
    mempool_pending: int = Field(description="Pending transactions")
    uncle_block_rate: Optional[float] = Field(None, description="Uncle block rate %")
    validator_participation: Optional[float] = Field(None, description="Validator participation %")
    mev_bundle_ratio: Optional[float] = Field(None, description="MEV bundle ratio %")
    last_update: str = Field(description="Last update timestamp")

class RecommendationResponse(BaseModel):
    """Complete recommendation response"""
    gas_fees: GasFeesResponse
    priority_fees: PriorityFeesResponse
    slippage: Optional[SlippageResponse] = None
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    source: str = Field(description="Recommendation source")
    latency_ms: float = Field(description="Response time in milliseconds")
    features_used: Optional[int] = Field(None, description="Number of features used")
    automation_level: Optional[str] = Field(default="fully_automated", description="Automation level")
    user_inputs_required: Optional[int] = Field(default=1, description="Number of user inputs required")

class AllRecommendationsResponse(BaseModel):
    """All three recommendation systems"""
    ultra_fast: RecommendationResponse
    cached_ml: Optional[RecommendationResponse]
    full_ai: RecommendationResponse
    network_status: NetworkStatus
    total_latency_ms: float = Field(description="Total response time")

# ===== FULLY AUTOMATED GAS FEE API =====

class AutomatedGasFeeAPI:
    """✅ FULLY AUTOMATED Gas Fee Estimation API"""
    
    def __init__(self):
        print("🚀 INITIALIZING FULLY AUTOMATED GAS FEE API")
        print("=" * 60)
        
        # Initialize components
        self.data_collector = EthereumDataCollector()
        self.feature_engineer = EthereumFeatureEngineer()
        self.fast_rules = FastRuleBasedRecommendations()
        self.cached_ml = CachedMLRecommendations()
        self.ai_pipeline = GasFeeCompletePipeline()
        
        # Real-time data
        self.current_network_data = {}
        self.last_update = 0
        self.running = False
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        print("✅ All systems initialized with FULL AUTOMATION 🤖")
        
    def start_background_services(self):
        """Start background data collection and ML services"""
        if self.running:
            return
            
        print("🔄 Starting background services...")
        self.running = True
        
        # Start cached ML background updates
        self.cached_ml.start_background_updates()
        
        # Start real-time data collection
        self._start_data_collection_thread()
        
        print("✅ Background services started")
    
    def stop_background_services(self):
        """Stop all background services"""
        print("🛑 Stopping background services...")
        self.running = False
        self.cached_ml.stop_background_updates()
        print("✅ Background services stopped")
    
    def _start_data_collection_thread(self):
        """Start background thread for real-time data collection"""
        def collect_data():
            while self.running:
                try:
                    # Get real-time network data
                    network_data = self.data_collector.get_enhanced_network_state()
                    
                    if network_data:
                        self.current_network_data = network_data
                        self.last_update = time.time()
                        
                        # Update cached ML with fresh data
                        self.cached_ml.update_real_time_cache(network_data)
                        
                        # Broadcast to WebSocket clients
                        asyncio.create_task(self._broadcast_network_update())
                    
                    time.sleep(2)  # Collect every 2 seconds
                    
                except Exception as e:
                    print(f"❌ Data collection error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    async def _broadcast_network_update(self):
        """Broadcast network updates to WebSocket clients"""
        if not self.websocket_connections:
            return
            
        network_status = self.get_network_status()
        message = {
            "type": "network_update",
            "data": network_status.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)
    
    def _get_automated_params(self, request) -> Dict:
        """✅ AUTOMATED: Extract and auto-fill all trade parameters"""
        try:
            # Get base trade size (only required parameter)
            trade_size_usd = getattr(request, 'trade_size_usd', 5000)
            token_address = getattr(request, 'token_address', None)
            
            # Use automated parameter collection for missing values
            automated_params = self.data_collector.get_fully_automated_params(
                trade_size_usd, token_address
            )
            
            # Override with any user-provided values
            if hasattr(request, 'pool_liquidity_usd') and request.pool_liquidity_usd is not None:
                automated_params['pool_liquidity_usd'] = request.pool_liquidity_usd
            
            if hasattr(request, 'volatility_score') and request.volatility_score is not None:
                automated_params['volatility_score'] = request.volatility_score
                
            if hasattr(request, 'user_urgency') and request.user_urgency is not None:
                automated_params['user_urgency'] = request.user_urgency
            
            if hasattr(request, 'base_fee') and request.base_fee is not None:
                automated_params['base_fee'] = request.base_fee
                
            if hasattr(request, 'network_util') and request.network_util is not None:
                automated_params['network_util'] = request.network_util
                
            if hasattr(request, 'mempool_size') and request.mempool_size is not None:
                automated_params['mempool_size'] = request.mempool_size
            
            print(f"🤖 Using automated params: liquidity=${automated_params['pool_liquidity_usd']:,.0f}, volatility={automated_params['volatility_score']:.2f}, urgency={automated_params['user_urgency']:.2f}")
            
            return automated_params
            
        except Exception as e:
            logger.error(f"Automated parameter extraction failed: {e}")
            # Fallback to basic automation
            network_data = self.current_network_data or self.data_collector.get_current_network_state()
            return {
                'base_fee': network_data.get('baseFeePerGas', 25e9) / 1e9,
                'network_util': network_data.get('network_utilization', 80.0),
                'mempool_size': network_data.get('mempool_pending_count', 150000),
                'trade_size_usd': trade_size_usd,
                'pool_liquidity_usd': 1000000,  # Default $1M
                'volatility_score': 0.5,        # Default medium
                'user_urgency': 0.5,            # Default medium
                'automation_source': 'fallback'
            }
    
    def get_network_status(self) -> NetworkStatus:
        """Get current network status"""
        if not self.current_network_data:
            raise HTTPException(status_code=503, detail="Network data not available yet")
        
        return NetworkStatus(
            block_number=self.current_network_data.get('blockNumber', 0),
            base_fee_gwei=self.current_network_data.get('baseFeePerGas', 0) / 1e9,
            network_utilization=self.current_network_data.get('network_utilization', 0),
            mempool_pending=self.current_network_data.get('mempool_pending_count', 0),
            uncle_block_rate=self.current_network_data.get('uncle_block_rate', 0) * 100 if 'uncle_block_rate' in self.current_network_data else None,
            validator_participation=self.current_network_data.get('validator_participation', 0) * 100 if 'validator_participation' in self.current_network_data else None,
            mev_bundle_ratio=self.current_network_data.get('flashbots_bundle_ratio', 0) * 100 if 'flashbots_bundle_ratio' in self.current_network_data else None,
            last_update=datetime.fromtimestamp(self.last_update).isoformat() if self.last_update else "Never"
        )
    
    def get_ultra_fast_recommendation(self, request) -> RecommendationResponse:
        """✅ AUTOMATED: Ultra-fast rule-based recommendation (~0.05ms)"""
        start_time = time.perf_counter()
        
        trade_params = self._get_automated_params(request)
        
        # Gas fees
        gas_fees = self.fast_rules.get_gas_fee_fast(
            trade_params['base_fee'],
            trade_params['network_util'], 
            trade_params['mempool_size']
        )
        
        # Priority fees
        priority_fees = self.fast_rules.get_priority_fee_fast(
            trade_params.get('mempool_congestion', trade_params['network_util']),
            trade_params['user_urgency']
        )
        
        # Slippage estimates
        slippage = self.fast_rules.get_slippage_fast(
            trade_params['trade_size_usd'],
            trade_params['pool_liquidity_usd'],
            trade_params['volatility_score']
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RecommendationResponse(
            gas_fees=GasFeesResponse(**gas_fees),
            priority_fees=PriorityFeesResponse(**priority_fees),
            slippage=SlippageResponse(**slippage),
            confidence=1.0,  # Rule-based is deterministic
            source="ultra_fast_automated",
            latency_ms=latency_ms,
            automation_level="fully_automated"
        )
    
    def get_cached_ml_recommendation(self, request) -> Optional[RecommendationResponse]:
        """✅ AUTOMATED: Hybrid cached ML recommendation (~1-2ms)"""
        start_time = time.perf_counter()
        
        trade_params = self._get_automated_params(request)
        cached_result = self.cached_ml.get_recommendation_fast(trade_params)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if not cached_result:
            return None
        
        return RecommendationResponse(
            gas_fees=GasFeesResponse(**cached_result['gas_fees']),
            priority_fees=PriorityFeesResponse(**cached_result['priority_fees']),
            slippage=SlippageResponse(**cached_result.get('slippage', {'conservative': 0.5, 'balanced': 0.3, 'aggressive': 0.1})),
            confidence=cached_result.get('confidence', 0.85),
            source=f"{cached_result.get('source', 'cached_ml')}_automated",
            latency_ms=latency_ms,
            automation_level="fully_automated"
        )
    
    def get_full_ai_recommendation(self, request) -> RecommendationResponse:
        """✅ AUTOMATED: Full AI recommendation with 80+ features"""
        start_time = time.perf_counter()
        
        trade_params = self._get_automated_params(request)
        ai_result = self.ai_pipeline.get_comprehensive_recommendation(trade_params)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return RecommendationResponse(
            gas_fees=GasFeesResponse(**ai_result['gas_fees']),
            priority_fees=PriorityFeesResponse(**ai_result['priority_fees']),
            slippage=SlippageResponse(**ai_result.get('slippage', {'conservative': 0.5, 'balanced': 0.3, 'aggressive': 0.1})),
            confidence=ai_result.get('confidence', 0.9),
            source=f"{ai_result.get('source', 'full_ai')}_automated",
            latency_ms=latency_ms,
            features_used=self.feature_engineer.get_total_feature_count(),
            automation_level="fully_automated"
        )
    
    def get_all_recommendations(self, request) -> AllRecommendationsResponse:
        """✅ AUTOMATED: Get all three recommendations"""
        total_start_time = time.perf_counter()
        
        # Get all recommendations
        ultra_fast = self.get_ultra_fast_recommendation(request)
        cached_ml = self.get_cached_ml_recommendation(request)
        full_ai = self.get_full_ai_recommendation(request)
        network_status = self.get_network_status()
        
        total_latency_ms = (time.perf_counter() - total_start_time) * 1000
        
        return AllRecommendationsResponse(
            ultra_fast=ultra_fast,
            cached_ml=cached_ml,
            full_ai=full_ai,
            network_status=network_status,
            total_latency_ms=total_latency_ms
        )

# ===== FASTAPI APPLICATION =====

# Initialize API
api = AutomatedGasFeeAPI()

# Create FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 API startup complete")
    api.start_background_services()
    yield
    # Shutdown
    print("🛑 API shutdown")
    api.stop_background_services()

app = FastAPI(
    title="⚡ FULLY AUTOMATED Gas Fee Estimation API 🤖",
    description="""
    ## 🚀 ZERO-CONFIG Gas Fee Prediction API
    
    **Just provide your trade size - everything else is automated!**
    
    ### 🤖 Full Automation Features:
    - ✅ **Pool Liquidity**: Auto-fetched from Uniswap, SushiSwap, 1inch APIs
    - ✅ **Volatility Score**: Auto-calculated from CoinGecko market data  
    - ✅ **User Urgency**: Auto-determined from network conditions
    - ✅ **Network Data**: Auto-collected from Ethereum mainnet
    
    ### ⚡ 3 Recommendation Systems:
    1. **Ultra-Fast Rule-based** (~0.05ms) - Simple heuristics for instant results
    2. **Hybrid Cached ML** (~1-2ms) - ML predictions with real-time adjustments  
    3. **Full AI Analysis** - Complete analysis using 80+ network features
    
    ### 🎯 Usage:
    ```json
    {
        "trade_size_usd": 10000
    }
    ```
    
    **That's it!** Everything else is automated.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AUTOMATED API ENDPOINTS =====

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "automation_level": "fully_automated",
        "network_data_available": bool(api.current_network_data),
        "last_update": datetime.fromtimestamp(api.last_update).isoformat() if api.last_update else None
    }

@app.get("/network/status", response_model=NetworkStatus)
async def get_network_status():
    """Get current Ethereum network status"""
    return api.get_network_status()

@app.post("/gas/ultra-fast", response_model=RecommendationResponse)
async def get_ultra_fast_gas_fees(request: AutomatedTradeRequest):
    """
    🤖 FULLY AUTOMATED Ultra-fast rule-based gas fee estimation (~0.05ms)
    
    **Only trade_size_usd required - everything else automated!**
    
    - ✅ Pool liquidity: Auto-fetched from DEX APIs
    - ✅ Volatility: Auto-calculated from market data
    - ✅ Urgency: Auto-determined from network conditions
    - ✅ Network data: Auto-collected in real-time
    
    Perfect for high-frequency trading when latency is critical.
    """
    return api.get_ultra_fast_recommendation(request)

@app.post("/gas/cached-ml", response_model=RecommendationResponse)
async def get_cached_ml_gas_fees(request: AutomatedTradeRequest):
    """
    🤖 FULLY AUTOMATED Hybrid cached ML gas fee estimation (~1-2ms)
    
    **Only trade_size_usd required - everything else automated!**
    
    Uses ML predictions cached in background with real-time micro-adjustments.
    Best balance of speed and accuracy for most trading applications.
    """
    cached_result = api.get_cached_ml_recommendation(request)
    if not cached_result:
        raise HTTPException(status_code=503, detail="ML cache not ready yet. Try again in a few minutes or use /gas/ultra-fast")
    return cached_result

@app.post("/gas/full-ai", response_model=RecommendationResponse)
async def get_full_ai_gas_fees(request: AutomatedTradeRequest):
    """
    🤖 FULLY AUTOMATED Complete AI gas fee analysis using 80+ network features
    
    **Only trade_size_usd required - everything else automated!**
    
    Most accurate recommendations using comprehensive network analysis.
    Best for large trades or when maximum accuracy is needed.
    """
    return api.get_full_ai_recommendation(request)

@app.post("/gas/all", response_model=AllRecommendationsResponse)
async def get_all_gas_fee_recommendations(request: AutomatedTradeRequest):
    """
    🤖 FULLY AUTOMATED Get recommendations from all three systems for comparison
    
    **Only trade_size_usd required - everything else automated!**
    
    Returns ultra-fast, cached ML, and full AI recommendations
    along with current network status.
    """
    return api.get_all_recommendations(request)

@app.post("/gas/automated", response_model=RecommendationResponse)
async def get_automated_gas_recommendation(request: AutomatedTradeRequest):
    """🤖 FULLY AUTOMATED - Zero user input gas fee recommendation"""
    try:
        start_time = time.time()
        
        auto_params = api.get_all_recommendations(request)
        
        return RecommendationResponse(
            status="success",
            data=auto_params,
            latency_ms=(time.time() - start_time) * 1000,
            automation=True,
            source="fully_automated"
        )
        
    except Exception as e:
        return RecommendationResponse(
            status="error",
            error=str(e),
            latency_ms=(time.time() - start_time) * 1000
        )

# ===== LEGACY COMPATIBILITY ENDPOINTS =====

@app.post("/gas/legacy/ultra-fast", response_model=RecommendationResponse)
async def get_legacy_ultra_fast_gas_fees(request: LegacyTradeRequest):
    """🔄 Legacy endpoint with backwards compatibility (all parameters optional)"""
    automated_request = AutomatedTradeRequest(**request.dict())
    return api.get_ultra_fast_recommendation(automated_request)

@app.post("/gas/legacy/cached-ml", response_model=RecommendationResponse)
async def get_legacy_cached_ml_gas_fees(request: LegacyTradeRequest):
    """🔄 Legacy endpoint with backwards compatibility (all parameters optional)"""
    automated_request = AutomatedTradeRequest(**request.dict())
    cached_result = api.get_cached_ml_recommendation(automated_request)
    if not cached_result:
        raise HTTPException(status_code=503, detail="ML cache not ready yet.")
    return cached_result

@app.post("/gas/legacy/full-ai", response_model=RecommendationResponse)
async def get_legacy_full_ai_gas_fees(request: LegacyTradeRequest):
    """🔄 Legacy endpoint with backwards compatibility (all parameters optional)"""
    automated_request = AutomatedTradeRequest(**request.dict())
    return api.get_full_ai_recommendation(automated_request)

@app.post("/gas/legacy/all", response_model=AllRecommendationsResponse)
async def get_legacy_all_gas_fee_recommendations(request: LegacyTradeRequest):
    """🔄 Legacy endpoint with backwards compatibility (all parameters optional)"""
    automated_request = AutomatedTradeRequest(**request.dict())
    return api.get_all_recommendations(automated_request)

# ===== WEBSOCKET AND UTILITY ENDPOINTS =====

@app.websocket("/gas/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gas fee updates
    
    Streams network status updates and can provide real-time recommendations.
    """
    await websocket.accept()
    api.websocket_connections.append(websocket)
    
    try:
        while True:
            # Wait for client messages
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "get_recommendations":
                # Client requesting recommendations
                params = data.get("params", {})
                trade_request = AutomatedTradeRequest(
                    trade_size_usd=params.get("trade_size_usd", 5000),
                    token_address=params.get("token_address")
                )
                recommendations = api.get_all_recommendations(trade_request)
                
                response = {
                    "type": "recommendations",
                    "data": recommendations.dict(),
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        api.websocket_connections.remove(websocket)

@app.get("/cache/status")
async def get_cache_status():
    """Get ML cache status for debugging"""
    cache_stats = api.cached_ml.get_cache_stats()
    
    return {
        "cache_ready": cache_stats['cached_predictions'] > 0,
        "cached_predictions": cache_stats['cached_predictions'],
        "last_update": datetime.fromtimestamp(cache_stats['last_update']).isoformat(),
        "cache_age_seconds": cache_stats['cache_age_seconds'],
        "is_valid": cache_stats['is_valid'],
        "background_running": cache_stats['background_running'],
        "cache_validity_limit": api.cached_ml.config.CACHE_VALIDITY_SECONDS,
        "automation_level": "fully_automated"
    }

@app.get("/automation/status")
async def get_automation_status():
    """Get automation system status"""
    return {
        "automation_level": "fully_automated",
        "liquidity_cache_size": len(api.data_collector.liquidity_cache),
        "volatility_cache_size": len(api.data_collector.volatility_cache),
        "dex_apis_available": list(api.data_collector.dex_apis.keys()),
        "network_data_fresh": (time.time() - api.last_update) < 60 if api.last_update else False,
        "user_inputs_required": 1,
        "parameters_automated": [
            "pool_liquidity_usd",
            "volatility_score", 
            "user_urgency",
            "base_fee",
            "network_utilization",
            "mempool_size"
        ]
    }

# ===== MAIN EXECUTION =====

def main():
    """Run the API server"""
    print("🚀 FULLY AUTOMATED ETHEREUM GAS FEE ESTIMATION API 🤖")
    print("=" * 70)
    print("🌐 Zero-config API server - just provide trade size!")
    print("📡 Real-time Ethereum data integration")
    print("⚡ 3 automated recommendation systems")
    print("🤖 Full parameter automation via DEX/Market APIs")
    print("🔗 WebSocket support for live updates")
    print("=" * 70)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )

if __name__ == "__main__":
    main() 