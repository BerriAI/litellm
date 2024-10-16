# import requests, json

# BASE_URL = 'http://localhost:8080'

# def test_hello_route():
#     data = {"model": "claude-instant-1", "messages": [{"role": "user", "content": "hey, how's it going?"}]}
#     headers = {'Content-Type': 'application/json'}
#     response = requests.get(BASE_URL, headers=headers, data=json.dumps(data))
#     print(response.text)
#     assert response.status_code == 200
#     print("Hello route test passed!")

# if __name__ == '__main__':
#     test_hello_route()
