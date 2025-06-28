"""
🚀 Quick Real-Time Data Test Runner
Run this to quickly verify all real-time data collection is working
"""

import sys
import os
import asyncio
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from comprehensive_data_test import ComprehensiveDataTest

def quick_realtime_test():
    """Quick test of essential real-time functions"""
    
    print("⚡ QUICK REAL-TIME DATA TEST")
    print("=" * 50)
    
    test_suite = ComprehensiveDataTest()
    
    # Test essential functions only
    essential_tests = [
        test_suite.test_01_connectivity,
        test_suite.test_02_current_network_state,
        test_suite.test_05_external_gas_estimates,
        test_suite.test_11_fully_automated_params,
        test_suite.test_15_performance_single_call
    ]
    
    for test_func in essential_tests:
        test_func()
        time.sleep(0.5)  # Small delay between tests
    
    # Quick summary
    passed = sum(1 for result in test_suite.test_results.values() if result['success'])
    total = len(test_suite.test_results)
    
    print("=" * 50)
    print(f"QUICK TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🟢 ALL ESSENTIAL FUNCTIONS WORKING!")
    elif passed >= total * 0.8:
        print("🟡 MOSTLY WORKING - Some minor issues")
    else:
        print("🔴 CRITICAL ISSUES - Check your configuration")

if __name__ == "__main__":
    quick_realtime_test() 