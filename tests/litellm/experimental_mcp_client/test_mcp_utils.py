import pytest
import requests
from unittest.mock import patch, mock_open
from litellm.experimental_mcp_client.mcp_utils import extract_json_from_markdown, fetch_mcp_servers, update_mcp_servers_file

def test_extract_json_from_markdown_valid():
    markdown_content = '''
    # Test Markdown
    Here's some JSON:
    ```json
    {
        "name": "test-server",
        "url": "http://test.com"
    }
    ```
    And another one:
    ```json
    {
        "name": "test-server-2",
        "url": "http://test2.com"
    }
    ```
    '''
    result = extract_json_from_markdown(markdown_content)
    assert len(result) == 2
    assert result[0]["name"] == "test-server"
    assert result[1]["name"] == "test-server-2"

def test_extract_json_from_markdown_invalid():
    markdown_content = '''
    # Test Markdown
    Here's some invalid JSON:
    ```json
    {
        "name": "test-server",
        invalid json here
    }
    ```
    And valid JSON:
    ```json
    {
        "name": "test-server-2",
        "url": "http://test2.com"
    }
    ```
    '''
    result = extract_json_from_markdown(markdown_content)
    assert len(result) == 1
    assert result[0]["name"] == "test-server-2"

def test_extract_json_from_markdown_empty():
    markdown_content = "# No JSON here"
    result = extract_json_from_markdown(markdown_content)
    assert len(result) == 0

@pytest.fixture
def mock_github_response():
    return [
        {"type": "dir", "name": "server1"},
        {"type": "file", "name": "something.txt"},
        {"type": "dir", "name": "server2"}
    ]

@pytest.fixture
def mock_readme_content():
    return '''
    # Server Config
    ```json
    {
        "name": "test-server",
        "url": "http://test.com"
    }
    ```
    '''
@pytest.fixture
def mock_mcp_readme_content():
    return '''
    # Server Config
    ```json
    {
        "mcpServers": {
            "brave-search": {
                "command": "docker",
                "args": [
                    "run",
                    "-i",
        "--rm",
        "-e",
        "BRAVE_API_KEY",
        "mcp/brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
    ```
    '''


@patch('requests.get')
def test_fetch_mcp_servers_success(mock_get, mock_github_response, mock_mcp_readme_content, mock_readme_content):
    # Mock the responses
    mock_get.side_effect = [
        type('Response', (), {
            'json': lambda: mock_github_response,
            'raise_for_status': lambda: None,
            'status_code': 200
        }),
        type('Response', (), {
            'text': mock_mcp_readme_content,
            'status_code': 200
        }),
        type('Response', (), {
            'text': mock_readme_content,
            'status_code': 200
        })
    ]
    
    result = fetch_mcp_servers()
    assert len(result) == 1
    assert all(isinstance(server, dict) for server in result)
    assert mock_get.call_count == 3  # One for base URL, two for READMEs

@patch('requests.get')
def test_fetch_mcp_servers_request_error(mock_get):
    mock_get.side_effect = requests.RequestException("Connection error")
    result = fetch_mcp_servers()
    assert result == []

@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
@patch('litellm.experimental_mcp_client.mcp_utils.fetch_mcp_servers')
def test_update_mcp_servers_file_success(mock_fetch, mock_json_dump, mock_file):
    mock_servers = [
        {"name": "server1", "url": "http://test1.com"},
        {"name": "server2", "url": "http://test2.com"}
    ]
    mock_fetch.return_value = mock_servers
    
    update_mcp_servers_file("test_output.json")
    
    mock_file.assert_called_once_with("test_output.json", 'w')
    mock_json_dump.assert_called_once_with(mock_servers, mock_file(), indent=2)

@patch('litellm.experimental_mcp_client.mcp_utils.fetch_mcp_servers')
def test_update_mcp_servers_file_no_servers(mock_fetch):
    mock_fetch.return_value = []
    update_mcp_servers_file("test_output.json")
    # No file should be written when no servers are fetched