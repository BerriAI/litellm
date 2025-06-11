# Test URL encoding handling for emails with + characters
import re
from urllib.parse import unquote

def test_user_id_parsing():
    # Simulate the raw query string that would come from the URL
    # When user calls: http://0.0.0.0:4000/user/info?user_id=machine-user+alp-air-admin-b58-b@tempus.com
    # The query string would be: user_id=machine-user+alp-air-admin-b58-b@tempus.com
    
    test_cases = [
        "user_id=machine-user+alp-air-admin-b58-b@tempus.com",
        "user_id=machine-user%2Balp-air-admin-b58-b@tempus.com",  # URL encoded +
        "user_id=regular@email.com",
        "user_id=test-user@domain.com&other_param=value"
    ]
    
    for query_string in test_cases:
        print(f"\nTesting query string: {query_string}")
        
        if 'user_id=' in query_string:
            match = re.search(r'user_id=([^&]*)', query_string)
            if match:
                raw_user_id = unquote(match.group(1))
                print(f"Extracted user_id: {raw_user_id}")
            else:
                print("No match found")
        else:
            print("user_id not found in query string")

if __name__ == "__main__":
    test_user_id_parsing() 