import sys
import os
sys.path.append('app')
try:
    from features import extract_features
    print("=== TESTING FEATURE EXTRACTION ===")
    test_event_1 = {
        'process_name': 'powershell.exe',
        'command_line': 'Get-Process',
        'user': 'SYSTEM',
        'timestamp': '2024-01-15T09:30:00Z',
        'remote_port': 0,
        'parent_process_name': 'services.exe'
    }
    
    print("\n1. Testing normal event:")
    result1 = extract_features(test_event_1)
    print(result1)
    
    test_event_2 = {
        'process_name': 'chrome.exe',
        'command_line': 'chrome.exe --url=https://example.com',
        'user': 'user123',
        'timestamp': '2024-01-15T20:45:00Z',
        'remote_port': 443,
        'parent_process_name': 'explorer.exe'
    }
    
    print("\n2. Testing network event:")
    result2 = extract_features(test_event_2)
    print(result2)
    
    
    test_event_3 = {
        'process_name': 'unknown.exe'
    }
    
    print("\n3. Testing minimal event:")
    result3 = extract_features(test_event_3)
    print(result3)
    
    print("\n=== ALL TESTS COMPLETED ===")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have pandas installed: pip install pandas")
except Exception as e:
    print(f"Error: {e}")