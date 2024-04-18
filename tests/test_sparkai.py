import os
import litellm
'''
os.environ['SPARKAI_APP_ID']='your_id'
os.environ['SPARKAI_API_KEY']='your_key'
os.environ['SPARKAI_API_SECRET']='your_secret_key'
os.environ['SPARKAI_DOMAIN']='your_domain_name'
os.environ['SPARKAI_URL']="your_url"
'''
os.environ['SPARKAI_APP_ID']='21628c31'
os.environ['SPARKAI_API_KEY']='b54eb223efc235aef46f061e5fd752ea'
os.environ['SPARKAI_API_SECRET']='MDIwNzM5MzVjOTcwMDI3MTdjYWNlZjE3'
os.environ['SPARKAI_DOMAIN']='generalv3.5'
os.environ['SPARKAI_URL']="wss://spark-openapi.cn-huabei-1.xf-yun.com/v3.2/chat"
def test_sparkai():
    response_1 = litellm.completion(
        model="spark_ai",
        messages=[{"content": "Hello, what is your name?", "role": "user"}],
        stream=False,
    )
    #print(response_1)
    response_1_text = response_1.choices[0].message.content
    response_1_type = response_1.choices[0].message.role
    print(response_1_text)
    print(response_1_type)

def test_spark_stream():
    response_2 = litellm.completion(
        model="spark_ai",
        messages=[{"content": "Hello, what is your name?", "role": "user"}],
        stream=True,
    )
    for response in response_2:
        print(response)


if __name__ == "__main__":
    test_spark_stream()
    test_sparkai()