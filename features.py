

import pandas as pd
from datetime import datetime

def extract_features(raw_event):
   
    try:
        print("Starting feature extraction...")
        
        features = {}
    
        print("Extracting process information...")
        
        process_name = raw_event.get('process_name', '')
        features['process_name_length'] = len(str(process_name))
        
        command_line = raw_event.get('command_line', '')
        features['command_line_length'] = len(str(command_line))
        print("Extracting user context...")
        
        user = str(raw_event.get('user', '')).lower()
        system_users = ['system', 'root', 'admin', 'administrator']
        features['is_system_user'] = 1 if user in system_users else 0
        print("Extracting time features...")
        
        event_time = raw_event.get('timestamp')
        if event_time:
            try:
                if isinstance(event_time, str):
                    if 'Z' in event_time:
                        event_time = event_time.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(event_time)
                else:
                    dt = event_time
                
                features['hour_of_day'] = dt.hour  
                features['day_of_week'] = dt.weekday()  
                features['is_working_hours'] = 1 if 9 <= dt.hour <= 17 else 0
                
            except Exception as time_error:
                print(f"Time conversion error: {time_error}")
                features['hour_of_day'] = 12 
                features['day_of_week'] = 0  
                features['is_working_hours'] = 1
        else:
    
            features['hour_of_day'] = 12
            features['day_of_week'] = 0
            features['is_working_hours'] = 1
        
        print("Extracting network features...")
        remote_port = raw_event.get('remote_port', 0)
        features['has_network_activity'] = 1 if int(remote_port) > 0 else 0
        common_ports = [80, 443, 22, 21, 25, 53, 110, 993, 995]
        features['is_common_port'] = 1 if int(remote_port) in common_ports else 0
        print("Extracting parent process features...")
        
        parent_process = raw_event.get('parent_process_name', '')
        features['parent_process_name_length'] = len(str(parent_process))
        print("Creating final feature set...")
        feature_order = [
            'process_name_length',
            'command_line_length', 
            'is_system_user',
            'hour_of_day',
            'day_of_week',
            'is_working_hours',
            'has_network_activity',
            'is_common_port',
            'parent_process_name_length'
        ]
        final_features = {}
        for feature_name in feature_order:
            if feature_name in features:
                final_features[feature_name] = features[feature_name]
            else:
                final_features[feature_name] = 0  
        features_df = pd.DataFrame([final_features])
        
        print("Feature extraction completed successfully!")
        print(f"Extracted {len(feature_order)} features")
        
        return features_df
        
    except Exception as e:
        print(f"ERROR in feature extraction: {e}")
        return create_default_features()

def create_default_features():
    print("Creating default features due to error...")
    
    default_features = {
        'process_name_length': 0,
        'command_line_length': 0,
        'is_system_user': 0,
        'hour_of_day': 12,
        'day_of_week': 0,
        'is_working_hours': 1,
        'has_network_activity': 0,
        'is_common_port': 0,
        'parent_process_name_length': 0
    }    
    return pd.DataFrame([default_features])
if __name__ == "__main__":
    print("Testing feature extraction...")
    sample_event = {
        'process_name': 'cmd.exe',
        'command_line': 'cmd.exe /c ping google.com',
        'user': 'ADMIN',
        'timestamp': '2024-01-15T14:30:00Z',
        'remote_port': 443,
        'parent_process_name': 'explorer.exe'
    }
    
    features = extract_features(sample_event)
    print("\n=== EXTRACTED FEATURES ===")
    print(features)
    print("\nTesting with minimal data...")
    minimal_event = {
        'process_name': 'test.exe',
        'user': 'john'
    }
    
    features2 = extract_features(minimal_event)
    print(features2)