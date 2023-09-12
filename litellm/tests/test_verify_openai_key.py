from litellm import verify_access_key

def bad_key():
    key = "bad-key"
    response = verify_access_key(key)
    if response == False:
        pass
    else:
        raise Exception("Bad key was not detected")
bad_key()