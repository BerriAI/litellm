import requests, json

BASE_URL = 'http://localhost:8080'

def test_hello_route():
    data = {"model": "ollama/llama2", "messages": [{"role": "user", "content": "whats 1+1"}]}
    headers = {'Content-Type': 'application/json'}
    response = requests.get(BASE_URL, headers=headers, data=json.dumps(data))
    print(response.text)
    assert response.status_code == 200
    assert ("two" in response.text) or ("2" in response.text)
    print("Hello route test passed!")

if __name__ == '__main__':
    test_hello_route()
