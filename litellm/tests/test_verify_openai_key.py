from litellm import verify_access_key

def bad_key():
    key = "bad-key"
    response = verify_access_key(key)
    print(f"response: {response}")
bad_key()