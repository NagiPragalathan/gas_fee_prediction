"""FastAPI application for Gas Fee Prediction System"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline import GasFeeCompletePipeline
from src.utils import validate_trade_parameters, format_gas_recommendation, health_check_system

# Initialize FastAPI app
app = FastAPI(
    title="Gas Fee Prediction API",
    description="Ultra-fast Ethereum gas fee predictions with ML and rule-based recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the gas fee predictor
gas_predictor = GasFeeCompletePipeline()

# Pydantic models for request/response
class TradeRequest(BaseModel):
    base_fee: float = Field(default=25.0, ge=0.1, le=1000.0, description="Current base fee in gwei")
    network_util: float = Field(default=80.0, ge=0.0, le=100.0, description="Network utilization percentage")
    mempool_size: int = Field(default=150000, ge=0, le=1000000, description="Number of pending transactions")
    trade_size_usd: float = Field(default=1000.0, ge=1.0, le=1000000000.0, description="Trade size in USD")
    pool_liquidity_usd: float = Field(default=1000000.0, ge=1000.0, description="Pool liquidity in USD")
    volatility_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Market volatility score (0-1)")
    user_urgency: float = Field(default=0.5, ge=0.0, le=1.0, description="User urgency level (0-1)")

class GasFeeResponse(BaseModel):
    status: str
    data: Dict
    timestamp: str
    latency_ms: float
    source: str
    confidence: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict
    recommendations: List[str]

class PerformanceTestRequest(BaseModel):
    num_requests: int = Field(default=100, ge=1, le=10000, description="Number of test requests")
    test_type: str = Field(default="comprehensive", description="Type of test to run")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Starting Gas Fee Prediction API")
    
    # Start background ML training
    gas_predictor.start_background_ml()
    
    # Train initial models if needed
    if not gas_predictor.gas_fee_models:
        print("🧠 Training initial ML models...")
        gas_predictor.train_gas_fee_models(hours_of_historical_data=24)
    
    print("✅ API ready for requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("⏹️ Shutting down Gas Fee Prediction API")
    gas_predictor.stop_full_system()

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Gas Fee Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "instant": "/api/v1/gas-recommendation",
            "comprehensive": "/api/v1/comprehensive-recommendation",
            "health": "/health",
            "network": "/api/v1/network-status",
            "performance": "/api/v1/performance-test"
        },
        "features": [
            "Ultra-fast rule-based recommendations (~0.05ms)",
            "Cached ML predictions with real-time adjustments (~0.1ms)",
            "Comprehensive ensemble predictions (~2ms)",
            "65+ feature engineering for ML models",
            "Background ML training every 10 seconds"
        ]
    }

@app.post("/api/v1/gas-recommendation", response_model=GasFeeResponse)
async def get_gas_recommendation(trade_request: TradeRequest):
    """
    Get instant gas fee recommendation (<2ms)
    
    This endpoint provides the fastest possible recommendation using:
    1. Cached ML predictions (if available)
    2. Ultra-fast rule-based fallback
    """
    start_time = time.time()
    
    try:
        # Validate and convert request
        trade_params = validate_trade_parameters(trade_request.dict())
        
        # Get instant recommendation
        recommendation = gas_predictor.get_instant_recommendation(trade_params)
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        
        return GasFeeResponse(
            status="success",
            data=recommendation,
            timestamp=datetime.now().isoformat(),
            latency_ms=total_latency,
            source=recommendation.get('source', 'unknown'),
            confidence=recommendation.get('confidence', 'medium')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/comprehensive-recommendation", response_model=GasFeeResponse)
async def get_comprehensive_recommendation(trade_request: TradeRequest):
    """
    Get comprehensive gas fee recommendation with ensemble approach
    
    This endpoint tries all available prediction methods and combines them
    for the highest accuracy possible.
    """
    start_time = time.time()
    
    try:
        # Validate and convert request
        trade_params = validate_trade_parameters(trade_request.dict())
        
        # Get comprehensive recommendation
        recommendation = gas_predictor.get_comprehensive_recommendation(trade_params)
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        
        return GasFeeResponse(
            status="success",
            data=recommendation,
            timestamp=datetime.now().isoformat(),
            latency_ms=total_latency,
            source=recommendation.get('source', 'ensemble'),
            confidence=recommendation.get('confidence', 'high')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive prediction failed: {str(e)}")

@app.get("/api/v1/network-status")
async def get_network_status():
    """Get current Ethereum network status"""
    try:
        network_data = gas_predictor.data_collector.get_current_network_state()
        
        # Add calculated fields for display
        network_data['base_fee_gwei'] = network_data['baseFeePerGas'] / 1e9
        network_data['congestion_level'] = (
            'high' if network_data['network_utilization'] > 90
            else 'medium' if network_data['network_utilization'] > 70
            else 'low'
        )
        
        return {
            "status": "success",
            "data": network_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network status unavailable: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    try:
        health_status = health_check_system(gas_predictor)
        
        return HealthResponse(
            status=health_status['overall_status'],
            timestamp=health_status['timestamp'],
            components=health_status['components'],
            recommendations=health_status['recommendations']
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)},
            recommendations=["System requires attention"]
        )

@app.post("/api/v1/performance-test")
async def run_performance_test(test_request: PerformanceTestRequest, background_tasks: BackgroundTasks):
    """
    Run performance test on the system
    
    This endpoint runs a performance test with the specified number of requests
    and returns latency statistics.
    """
    try:
        print(f"🚀 Running performance test with {test_request.num_requests} requests")
        
        # Run the performance test
        if test_request.test_type == "comprehensive":
            performance_stats = gas_predictor.run_performance_test(test_request.num_requests)
        else:
            # Run specific test type
            performance_stats = gas_predictor.run_performance_test(100)  # Default for now
        
        return {
            "status": "success",
            "test_type": test_request.test_type,
            "num_requests": test_request.num_requests,
            "results": performance_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")

@app.get("/api/v1/cache-stats")
async def get_cache_stats():
    """Get ML cache statistics"""
    try:
        cache_stats = gas_predictor.cached_ml.get_cache_stats()
        
        return {
            "status": "success",
            "data": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats unavailable: {str(e)}")

@app.post("/api/v1/scenarios/demo")
async def run_demo_scenarios():
    """Run demonstration scenarios"""
    try:
        # Run demo scenarios
        gas_predictor.demo_all_scenarios()
        
        return {
            "status": "success",
            "message": "Demo scenarios completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo scenarios failed: {str(e)}")

@app.post("/api/v1/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining in background"""
    try:
        # Add retraining to background tasks
        background_tasks.add_task(
            gas_predictor.train_gas_fee_models,
            hours_of_historical_data=168  # 7 days
        )
        
        return {
            "status": "success",
            "message": "Model retraining started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "/api/v1/gas-recommendation",
            "/api/v1/comprehensive-recommendation",
            "/api/v1/network-status",
            "/health"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 