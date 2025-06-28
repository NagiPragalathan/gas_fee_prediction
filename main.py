"""Main execution script for Gas Fee Prediction System"""

import argparse
import sys
import os
import time
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import GasFeeCompletePipeline
from src.utils import create_sample_trade_scenarios, format_gas_recommendation, export_performance_report

def run_quick_demo():
    """Run a quick demonstration of the system"""
    print("🚀 Gas Fee Prediction System - Quick Demo")
    print("=" * 60)
    
    # Create predictor
    predictor = GasFeeCompletePipeline()
    
    # Sample request
    trade_params = {
        'base_fee': 25.0,
        'network_util': 80.0,
        'mempool_size': 150000,
        'trade_size_usd': 5000,
        'pool_liquidity_usd': 1000000,
        'volatility_score': 0.5,
        'user_urgency': 0.7
    }
    
    print("📥 Sample Trade Parameters:")
    for key, value in trade_params.items():
        print(f"   {key}: {value}")
    
    # Get recommendation
    start_time = time.time()
    recommendation = predictor.get_instant_recommendation(trade_params)
    total_time = (time.time() - start_time) * 1000
    
    print(f"\n📤 Recommendation (Generated in {total_time:.2f}ms):")
    print(format_gas_recommendation(recommendation))
    
    print("\n✅ Quick demo completed!")

def run_full_demo():
    """Run full system demonstration"""
    print("🚀 Gas Fee Prediction System - Full Demo")
    print("=" * 60)
    
    # Create predictor
    predictor = GasFeeCompletePipeline()
    
    try:
        # Start full system
        if predictor.start_full_system():
            print("\n💤 System running full demo...")
            
            # Run 5 sample requests
            scenarios = create_sample_trade_scenarios()
            
            for i, scenario in enumerate(scenarios[:3], 1):  # First 3 scenarios
                print(f"\n🔄 Demo Request #{i}: {scenario['name']}")
                
                start_time = time.time()
                recommendation = predictor.get_comprehensive_recommendation(scenario['params'])
                total_time = (time.time() - start_time) * 1000
                
                print(f"📤 Generated in {total_time:.2f}ms:")
                print(f"   Gas (standard): {recommendation['gas_fees']['standard']:.2f} gwei")
                print(f"   Priority (medium): {recommendation['priority_fees']['medium']:.2f} gwei")
                print(f"   Source: {recommendation.get('source', 'unknown')}")
                
                time.sleep(1)  # Brief pause between requests
            
            print("\n✅ Full demo completed!")
        
        else:
            print("❌ Failed to start system")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    finally:
        predictor.stop_full_system()

def run_performance_test(num_requests=1000):
    """Run performance test"""
    print(f"🚀 Running Performance Test ({num_requests:,} requests)")
    print("=" * 60)
    
    predictor = GasFeeCompletePipeline()
    
    # Run performance test
    performance_stats = predictor.run_performance_test(num_requests)
    
    # Export results
    report = export_performance_report(performance_stats)
    print("\n📊 Performance Test Completed!")
    print(report)

def run_training_mode(hours_back=168):
    """Run model training"""
    print(f"🧠 Training Gas Fee Models ({hours_back} hours of data)")
    print("=" * 60)
    
    predictor = GasFeeCompletePipeline()
    
    # Train models
    predictor.train_gas_fee_models(hours_of_historical_data=hours_back)
    
    print("✅ Model training completed!")

def run_api_server():
    """Start the FastAPI server"""
    print("🌐 Starting Gas Fee Prediction API Server")
    print("=" * 60)
    
    try:
        import uvicorn
        from api.gas_fee_api import app
        
        print("📡 Server starting on http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔄 Health Check: http://localhost:8000/health")
        print("\nPress Ctrl+C to stop the server")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")

def interactive_mode():
    """Run interactive mode for manual testing"""
    print("🎮 Gas Fee Prediction - Interactive Mode")
    print("=" * 60)
    
    predictor = GasFeeCompletePipeline()
    
    # Start background ML
    predictor.start_background_ml()
    
    try:
        while True:
            print("\n📋 Enter trade parameters (or 'quit' to exit):")
            
            try:
                base_fee = float(input("Base fee (gwei) [25.0]: ") or "25.0")
                network_util = float(input("Network utilization (%) [80.0]: ") or "80.0")
                mempool_size = int(input("Mempool size [150000]: ") or "150000")
                trade_size = float(input("Trade size (USD) [1000]: ") or "1000")
                user_urgency = float(input("User urgency (0-1) [0.5]: ") or "0.5")
                
                trade_params = {
                    'base_fee': base_fee,
                    'network_util': network_util,
                    'mempool_size': mempool_size,
                    'trade_size_usd': trade_size,
                    'pool_liquidity_usd': 1000000,
                    'volatility_score': 0.5,
                    'user_urgency': user_urgency
                }
                
                # Get recommendation
                start_time = time.time()
                recommendation = predictor.get_comprehensive_recommendation(trade_params)
                total_time = (time.time() - start_time) * 1000
                
                print(f"\n📤 Recommendation (Generated in {total_time:.2f}ms):")
                print(format_gas_recommendation(recommendation))
                
            except ValueError:
                print("❌ Invalid input. Please enter numeric values.")
            except KeyboardInterrupt:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        predictor.stop_background_ml()
        print("\n👋 Interactive mode ended")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gas Fee Prediction System")
    parser.add_argument('mode', choices=[
        'quick-demo', 'full-demo', 'performance', 'train', 'api', 'interactive'
    ], help='Execution mode')
    parser.add_argument('--requests', type=int, default=1000, 
                       help='Number of requests for performance test')
    parser.add_argument('--hours', type=int, default=168,
                       help='Hours of historical data for training')
    
    args = parser.parse_args()
    
    print(f"🚀 Gas Fee Prediction System")
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    if args.mode == 'quick-demo':
        run_quick_demo()
    elif args.mode == 'full-demo':
        run_full_demo()
    elif args.mode == 'performance':
        run_performance_test(args.requests)
    elif args.mode == 'train':
        run_training_mode(args.hours)
    elif args.mode == 'api':
        run_api_server()
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        print("❌ Invalid mode selected")
        parser.print_help()

if __name__ == "__main__":
    main() 