"""Quick tests for individual endpoints"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Quick health check"""
    print("🏥 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_network_status():
    """Quick network status check"""
    print("📊 Testing Network Status...")
    response = requests.get(f"{BASE_URL}/network/status")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_cache_status():
    """Quick cache status check"""
    print("💾 Testing Cache Status...")
    response = requests.get(f"{BASE_URL}/cache/status")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_ultra_fast():
    """Quick ultra-fast test"""
    print("🏃‍♂️ Testing Ultra-Fast...")
    params = {"trade_size_usd": 5000, "user_urgency": 0.7}
    response = requests.post(f"{BASE_URL}/gas/ultra-fast", json=params)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Latency: {data['latency_ms']:.3f}ms")
        print(f"Standard Gas: {data['gas_fees']['standard']:.2f} gwei")
        print(f"Medium Priority: {data['priority_fees']['medium']:.2f} gwei")
    else:
        print(response.text)
    print()

def test_cached_ml():
    """Quick cached ML test"""
    print("🧠 Testing Cached ML...")
    params = {"trade_size_usd": 10000, "user_urgency": 0.8}
    response = requests.post(f"{BASE_URL}/gas/cached-ml", json=params)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Latency: {data['latency_ms']:.3f}ms")
        print(f"Standard Gas: {data['gas_fees']['standard']:.2f} gwei")
        print(f"Confidence: {data.get('confidence', 'N/A')}")
    elif response.status_code == 503:
        print("Cache not ready yet")
    else:
        print(response.text)
    print()

def test_full_ai():
    """Quick full AI test"""
    print("🤖 Testing Full AI...")
    params = {"trade_size_usd": 50000, "user_urgency": 0.9}
    response = requests.post(f"{BASE_URL}/gas/full-ai", json=params)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Latency: {data['latency_ms']:.1f}ms")
        print(f"Features Used: {data.get('features_used', 'N/A')}")
        print(f"Standard Gas: {data['gas_fees']['standard']:.2f} gwei")
    else:
        print(response.text)
    print()

def test_all_recommendations():
    """Quick all recommendations test"""
    print("🚀 Testing All Recommendations...")
    params = {"trade_size_usd": 15000, "user_urgency": 0.6}
    response = requests.post(f"{BASE_URL}/gas/all", json=params)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Latency: {data['total_latency_ms']:.1f}ms")
        print(f"Ultra-fast: {data['ultra_fast']['gas_fees']['standard']:.2f} gwei")
        if data['cached_ml']:
            print(f"Cached ML: {data['cached_ml']['gas_fees']['standard']:.2f} gwei")
        print(f"Full AI: {data['full_ai']['gas_fees']['standard']:.2f} gwei")
    else:
        print(response.text)
    print()

if __name__ == "__main__":
    print("🧪 QUICK ENDPOINT TESTS")
    print("=" * 40)
    
    test_health()
    test_network_status()
    test_cache_status()
    test_ultra_fast()
    test_cached_ml()
    test_full_ai()
    test_all_recommendations()
    
    print("✅ Quick tests completed!") 