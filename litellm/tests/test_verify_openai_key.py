from litellm import verify_access_key
import os

def test_bad_key():
    key = "bad-key"
    response = verify_access_key(key)
    if response == False:
        pass
    else:
        raise Exception("Bad key was not detected")
test_bad_key()

def test_good_key():
    key = os.environ['OPENAI_API_KEY']
    response = verify_access_key(key)
    if response == True:
        pass
    else:
        raise Exception("Good key did not pass")
test_good_key()