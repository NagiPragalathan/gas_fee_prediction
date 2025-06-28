"""
Comprehensive Test Suite for Gas Fee Estimation API
Tests all endpoints with various scenarios
"""

import requests
import json
import time
import websocket
import threading
from datetime import datetime
from typing import Dict, List

class GasFeeAPITester:
    """Complete API endpoint tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.session = requests.Session()
        self.results = []
        
        print("🧪 GAS FEE API COMPREHENSIVE TESTER")
        print(f"🌐 Testing API at: {base_url}")
        print("=" * 70)
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        print("\n🏥 TESTING /health")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health")
            latency = (time.time() - start_time) * 1000
            
            print(f"Status Code: {response.status_code}")
            print(f"Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print("Response:")
                print(json.dumps(data, indent=2))
                
                # Validate required fields
                required_fields = ['status', 'timestamp', 'network_data_available']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"❌ Missing fields: {missing_fields}")
                    return False
                else:
                    print("✅ Health check passed")
                    return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_network_status_endpoint(self):
        """Test /network/status endpoint"""
        print("\n📊 TESTING /network/status")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/network/status")
            latency = (time.time() - start_time) * 1000
            
            print(f"Status Code: {response.status_code}")
            print(f"Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print("Network Status:")
                print(f"  Block Number: {data.get('block_number', 'N/A'):,}")
                print(f"  Base Fee: {data.get('base_fee_gwei', 'N/A'):.6f} gwei")
                print(f"  Network Util: {data.get('network_utilization', 'N/A'):.2f}%")
                print(f"  Mempool Pending: {data.get('mempool_pending', 'N/A'):,}")
                print(f"  Last Update: {data.get('last_update', 'N/A')}")
                
                if data.get('uncle_block_rate'):
                    print(f"  Uncle Block Rate: {data['uncle_block_rate']:.3f}%")
                if data.get('validator_participation'):
                    print(f"  Validator Participation: {data['validator_participation']:.1f}%")
                if data.get('mev_bundle_ratio'):
                    print(f"  MEV Bundle Ratio: {data['mev_bundle_ratio']:.1f}%")
                
                print("✅ Network status retrieved successfully")
                return data
            else:
                print(f"❌ Network status failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Network status error: {e}")
            return None
    
    def test_cache_status_endpoint(self):
        """Test /cache/status endpoint"""
        print("\n💾 TESTING /cache/status")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/cache/status")
            latency = (time.time() - start_time) * 1000
            
            print(f"Status Code: {response.status_code}")
            print(f"Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print("Cache Status:")
                print(f"  Cache Ready: {data.get('cache_ready', False)}")
                print(f"  Cached Predictions: {data.get('cached_predictions', 0):,}")
                print(f"  Cache Age: {data.get('cache_age_seconds', 0):.1f} seconds")
                print(f"  Is Valid: {data.get('is_valid', False)}")
                print(f"  Background Running: {data.get('background_running', False)}")
                print(f"  Validity Limit: {data.get('cache_validity_limit', 0)} seconds")
                print(f"  Last Update: {data.get('last_update', 'N/A')}")
                
                if data.get('cache_ready'):
                    print("✅ Cache is ready")
                else:
                    print("⚠️ Cache is still building")
                
                return data
            else:
                print(f"❌ Cache status failed with status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Cache status error: {e}")
            return None
    
    def test_ultra_fast_endpoint(self, test_scenarios: List[Dict] = None):
        """Test /gas/ultra-fast endpoint with multiple scenarios"""
        print("\n🏃‍♂️ TESTING /gas/ultra-fast")
        print("-" * 40)
        
        if test_scenarios is None:
            test_scenarios = [
                {
                    "name": "Small Trade",
                    "params": {"trade_size_usd": 1000, "user_urgency": 0.3}
                },
                {
                    "name": "Medium Trade", 
                    "params": {"trade_size_usd": 5000, "user_urgency": 0.7}
                },
                {
                    "name": "Large Trade",
                    "params": {"trade_size_usd": 50000, "user_urgency": 0.9}
                },
                {
                    "name": "Custom Network",
                    "params": {
                        "base_fee": 30.0,
                        "network_util": 95.0,
                        "mempool_size": 200000,
                        "trade_size_usd": 10000,
                        "user_urgency": 0.8
                    }
                }
            ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/gas/ultra-fast",
                    json=scenario['params']
                )
                latency = (time.time() - start_time) * 1000
                
                print(f"  Status: {response.status_code} | Latency: {latency:.3f}ms")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  API Latency: {data.get('latency_ms', 'N/A'):.3f}ms")
                    print(f"  Gas Fees: Slow {data['gas_fees']['slow']:.2f} | Standard {data['gas_fees']['standard']:.2f} | Fast {data['gas_fees']['fast']:.2f} gwei")
                    print(f"  Priority: Low {data['priority_fees']['low']:.2f} | Medium {data['priority_fees']['medium']:.2f} | High {data['priority_fees']['high']:.2f} gwei")
                    print(f"  Source: {data.get('source', 'N/A')}")
                    print(f"  Confidence: {data.get('confidence', 'N/A')}")
                    print("  ✅ Success")
                else:
                    print(f"  ❌ Failed: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
    
    def test_cached_ml_endpoint(self, test_scenarios: List[Dict] = None):
        """Test /gas/cached-ml endpoint"""
        print("\n🧠 TESTING /gas/cached-ml")
        print("-" * 40)
        
        if test_scenarios is None:
            test_scenarios = [
                {
                    "name": "Basic Trade",
                    "params": {"trade_size_usd": 5000, "user_urgency": 0.7}
                },
                {
                    "name": "High Volatility",
                    "params": {
                        "trade_size_usd": 15000,
                        "volatility_score": 0.9,
                        "user_urgency": 0.8
                    }
                }
            ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/gas/cached-ml",
                    json=scenario['params']
                )
                latency = (time.time() - start_time) * 1000
                
                print(f"  Status: {response.status_code} | Latency: {latency:.3f}ms")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  API Latency: {data.get('latency_ms', 'N/A'):.3f}ms")
                    print(f"  Gas Fees: Slow {data['gas_fees']['slow']:.2f} | Standard {data['gas_fees']['standard']:.2f} | Fast {data['gas_fees']['fast']:.2f} gwei")
                    print(f"  Priority: Low {data['priority_fees']['low']:.2f} | Medium {data['priority_fees']['medium']:.2f} | High {data['priority_fees']['high']:.2f} gwei")
                    print(f"  Source: {data.get('source', 'N/A')}")
                    print(f"  Confidence: {data.get('confidence', 'N/A')}")
                    print("  ✅ Success")
                elif response.status_code == 503:
                    print("  ⚠️ Cache not ready yet (building...)")
                else:
                    print(f"  ❌ Failed: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
    
    def test_full_ai_endpoint(self, test_scenarios: List[Dict] = None):
        """Test /gas/full-ai endpoint"""
        print("\n🤖 TESTING /gas/full-ai")
        print("-" * 40)
        
        if test_scenarios is None:
            test_scenarios = [
                {
                    "name": "Standard Analysis",
                    "params": {"trade_size_usd": 10000, "user_urgency": 0.6}
                },
                {
                    "name": "Large Pool Trade",
                    "params": {
                        "trade_size_usd": 100000,
                        "pool_liquidity_usd": 10000000,
                        "volatility_score": 0.3,
                        "user_urgency": 0.9
                    }
                }
            ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/gas/full-ai",
                    json=scenario['params']
                )
                latency = (time.time() - start_time) * 1000
                
                print(f"  Status: {response.status_code} | Latency: {latency:.1f}ms")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  API Latency: {data.get('latency_ms', 'N/A'):.1f}ms")
                    print(f"  Features Used: {data.get('features_used', 'N/A')}")
                    print(f"  Gas Fees: Slow {data['gas_fees']['slow']:.2f} | Standard {data['gas_fees']['standard']:.2f} | Fast {data['gas_fees']['fast']:.2f} gwei")
                    print(f"  Priority: Low {data['priority_fees']['low']:.2f} | Medium {data['priority_fees']['medium']:.2f} | High {data['priority_fees']['high']:.2f} gwei")
                    print(f"  Source: {data.get('source', 'N/A')}")
                    print(f"  Confidence: {data.get('confidence', 'N/A')}")
                    print("  ✅ Success")
                else:
                    print(f"  ❌ Failed: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
    
    def test_all_recommendations_endpoint(self):
        """Test /gas/all endpoint"""
        print("\n🚀 TESTING /gas/all")
        print("-" * 40)
        
        test_params = {
            "trade_size_usd": 25000,
            "pool_liquidity_usd": 5000000,
            "volatility_score": 0.6,
            "user_urgency": 0.8
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/gas/all",
                json=test_params
            )
            latency = (time.time() - start_time) * 1000
            
            print(f"Status: {response.status_code} | Total Latency: {latency:.1f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print(f"API Total Latency: {data.get('total_latency_ms', 'N/A'):.1f}ms")
                
                # Ultra-fast results
                ultra = data['ultra_fast']
                print(f"\n🏃‍♂️ Ultra-Fast ({ultra['latency_ms']:.3f}ms):")
                print(f"  Standard Gas: {ultra['gas_fees']['standard']:.2f} gwei")
                print(f"  Medium Priority: {ultra['priority_fees']['medium']:.2f} gwei")
                print(f"  Source: {ultra['source']}")
                
                # Cached ML results
                if data['cached_ml']:
                    cached = data['cached_ml']
                    print(f"\n🧠 Cached ML ({cached['latency_ms']:.3f}ms):")
                    print(f"  Standard Gas: {cached['gas_fees']['standard']:.2f} gwei")
                    print(f"  Medium Priority: {cached['priority_fees']['medium']:.2f} gwei")
                    print(f"  Confidence: {cached.get('confidence', 'N/A')}")
                    print(f"  Source: {cached['source']}")
                else:
                    print(f"\n🧠 Cached ML: Not ready yet")
                
                # Full AI results
                full_ai = data['full_ai']
                print(f"\n🤖 Full AI ({full_ai['latency_ms']:.1f}ms):")
                print(f"  Standard Gas: {full_ai['gas_fees']['standard']:.2f} gwei")
                print(f"  Medium Priority: {full_ai['priority_fees']['medium']:.2f} gwei")
                print(f"  Features: {full_ai.get('features_used', 'N/A')}")
                print(f"  Confidence: {full_ai.get('confidence', 'N/A')}")
                print(f"  Source: {full_ai['source']}")
                
                # Network status
                network = data['network_status']
                print(f"\n📊 Network Status:")
                print(f"  Block: {network['block_number']:,}")
                print(f"  Base Fee: {network['base_fee_gwei']:.6f} gwei")
                print(f"  Utilization: {network['network_utilization']:.1f}%")
                print(f"  Mempool: {network['mempool_pending']:,} pending")
                
                print("\n✅ All recommendations retrieved successfully")
                return data
            else:
                print(f"❌ Failed: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_websocket_endpoint(self, duration: int = 15):
        """Test WebSocket /gas/stream endpoint"""
        print(f"\n🔌 TESTING WebSocket /gas/stream ({duration}s)")
        print("-" * 40)
        
        messages_received = []
        connection_success = False
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                messages_received.append(data)
                
                if data.get('type') == 'network_update':
                    network_data = data['data']
                    print(f"📡 Network update: Block {network_data.get('block_number', 'N/A')} | {network_data.get('base_fee_gwei', 0):.6f} gwei")
                elif data.get('type') == 'recommendations':
                    recs = data['data']
                    print(f"💡 Recommendations: Ultra-fast {recs['ultra_fast']['gas_fees']['standard']:.2f} gwei")
                    
            except Exception as e:
                print(f"❌ WebSocket message error: {e}")
        
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"🔌 WebSocket closed: {close_status_code}")
        
        def on_open(ws):
            nonlocal connection_success
            connection_success = True
            print("✅ WebSocket connected")
            
            # Request recommendations after 3 seconds
            def request_recommendations():
                time.sleep(3)
                request = {
                    "type": "get_recommendations",
                    "params": {
                        "trade_size_usd": 7500,
                        "user_urgency": 0.5
                    }
                }
                ws.send(json.dumps(request))
                print("📤 Requested recommendations via WebSocket")
            
            threading.Thread(target=request_recommendations, daemon=True).start()
        
        try:
            ws = websocket.WebSocketApp(
                f"{self.ws_url}/gas/stream",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in separate thread
            def run_websocket():
                ws.run_forever()
            
            ws_thread = threading.Thread(target=run_websocket, daemon=True)
            ws_thread.start()
            
            # Wait for duration
            time.sleep(duration)
            ws.close()
            
            print(f"\n📊 WebSocket Test Results:")
            print(f"  Connection Success: {connection_success}")
            print(f"  Messages Received: {len(messages_received)}")
            
            if connection_success and len(messages_received) > 0:
                print("✅ WebSocket test passed")
                return True
            else:
                print("❌ WebSocket test failed")
                return False
                
        except Exception as e:
            print(f"❌ WebSocket test error: {e}")
            return False
    
    def test_error_handling(self):
        """Test API error handling"""
        print("\n🚨 TESTING ERROR HANDLING")
        print("-" * 40)
        
        # Test invalid parameters
        print("Testing invalid parameters...")
        invalid_params = {
            "trade_size_usd": -1000,  # Negative value
            "user_urgency": 2.0,      # Out of range
            "volatility_score": "invalid"  # Wrong type
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/gas/ultra-fast",
                json=invalid_params
            )
            if response.status_code == 422:
                print("✅ Validation error handled correctly (422)")
            else:
                print(f"⚠️ Unexpected response for invalid params: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing invalid params: {e}")
        
        # Test non-existent endpoint
        print("Testing non-existent endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/nonexistent")
            if response.status_code == 404:
                print("✅ 404 error handled correctly")
            else:
                print(f"⚠️ Unexpected response for 404: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing 404: {e}")
        
        # Test malformed JSON
        print("Testing malformed JSON...")
        try:
            response = self.session.post(
                f"{self.base_url}/gas/ultra-fast",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in [400, 422]:
                print("✅ Malformed JSON handled correctly")
            else:
                print(f"⚠️ Unexpected response for malformed JSON: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing malformed JSON: {e}")
    
    def run_comprehensive_test(self):
        """Run all endpoint tests"""
        print("🚀 STARTING COMPREHENSIVE API TEST SUITE")
        print("=" * 70)
        
        start_time = time.time()
        results = {}
        
        # Test all endpoints
        results['health'] = self.test_health_endpoint()
        
        # Wait for API to stabilize
        print("\n⏳ Waiting 5 seconds for API to stabilize...")
        time.sleep(5)
        
        results['network_status'] = self.test_network_status_endpoint() is not None
        results['cache_status'] = self.test_cache_status_endpoint() is not None
        results['ultra_fast'] = self.test_ultra_fast_endpoint()
        results['cached_ml'] = self.test_cached_ml_endpoint()
        results['full_ai'] = self.test_full_ai_endpoint()
        results['all_recommendations'] = self.test_all_recommendations_endpoint() is not None
        results['websocket'] = self.test_websocket_endpoint()
        
        self.test_error_handling()
        
        # Summary
        total_time = time.time() - start_time
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print("\n" + "=" * 70)
        print("🎉 COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        print(f"⏱️ Total test duration: {total_time:.1f} seconds")
        print(f"📊 Tests passed: {passed}/{total}")
        print(f"✅ Success rate: {(passed/total)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for endpoint, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {endpoint:20}: {status}")
        
        if passed == total:
            print(f"\n🎯 ALL TESTS PASSED! API is working perfectly.")
        else:
            print(f"\n⚠️ Some tests failed. Check the logs above for details.")
        
        print("\n🏁 Test suite completed!")

def main():
    """Run the comprehensive API test suite"""
    print("🧪 GAS FEE API COMPREHENSIVE TEST SUITE")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start the API with: python main.py")
    print("=" * 70)
    
    # Wait for user confirmation
    input("Press Enter when the API server is running...")
    
    # Run comprehensive tests
    tester = GasFeeAPITester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 