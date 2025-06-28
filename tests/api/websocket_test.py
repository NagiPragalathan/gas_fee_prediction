"""WebSocket endpoint test"""

import asyncio
import websockets
import json
import time

async def test_websocket():
    """Test WebSocket connection and functionality"""
    uri = "ws://localhost:8000/gas/stream"
    
    print("🔌 Testing WebSocket Connection...")
    print(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send a request for recommendations
            request = {
                "type": "get_recommendations",
                "params": {
                    "trade_size_usd": 10000,
                    "user_urgency": 0.8,
                    "volatility_score": 0.6
                }
            }
            
            await websocket.send(json.dumps(request))
            print("📤 Sent recommendation request")
            
            # Listen for messages for 20 seconds
            message_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 20:  # 20 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\n📨 Message {message_count}: {data['type']}")
                    print(f"Timestamp: {data['timestamp']}")
                    
                    if data['type'] == 'network_update':
                        network = data['data']
                        print(f"Block: {network.get('block_number', 'N/A'):,}")
                        print(f"Base Fee: {network.get('base_fee_gwei', 0):.6f} gwei")
                        print(f"Network Util: {network.get('network_utilization', 0):.1f}%")
                        
                    elif data['type'] == 'recommendations':
                        recs = data['data']
                        print(f"Total Latency: {recs.get('total_latency_ms', 0):.1f}ms")
                        print(f"Ultra-fast Standard: {recs['ultra_fast']['gas_fees']['standard']:.2f} gwei")
                        if recs.get('cached_ml'):
                            print(f"Cached ML Standard: {recs['cached_ml']['gas_fees']['standard']:.2f} gwei")
                        print(f"Full AI Standard: {recs['full_ai']['gas_fees']['standard']:.2f} gwei")
                    
                except asyncio.TimeoutError:
                    print(".", end="", flush=True)  # Show we're still listening
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\n🔌 WebSocket connection closed")
                    break
            
            print(f"\n\n📊 WebSocket Test Summary:")
            print(f"Messages received: {message_count}")
            print(f"Test duration: {time.time() - start_time:.1f} seconds")
            
            if message_count > 0:
                print("✅ WebSocket test passed")
            else:
                print("❌ No messages received")
                
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")

if __name__ == "__main__":
    print("🔌 WEBSOCKET TEST")
    print("=" * 40)
    asyncio.run(test_websocket()) 