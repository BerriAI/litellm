"""
Utility functions for Model Context Protocol (MCP) server management.
"""
import os
import json
import requests
import re

from typing import Dict, Any, List

def extract_json_from_markdown(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extracts JSON configuration from markdown content by looking for code blocks.
    
    Args:
        markdown_content (str): The markdown content to parse
        
    Returns:
        Dict[str, Any]: Extracted JSON configuration or empty dict if not found
    """
    # Use a regex to find all JSON code blocks
    json_blocks = re.findall(r"```json\s*([\s\S]*?)```", markdown_content, re.DOTALL)
    extracted_jsons = []
    for block in json_blocks:
        try:
            # Attempt to parse each JSON block
            extracted_jsons.append(json.loads(block.strip()))
        except json.JSONDecodeError:
            continue  # Skip invalid JSON blocks
    
    return extracted_jsons

def fetch_mcp_servers() -> List[Dict[str, Any]]:
    """
    Fetches MCP server configurations from the modelcontextprotocol/servers repository
    and returns them in a standardized format.
    Scans the src directory README.md files for server configurations and extracts JSON from key "mcpServers".
    
    Returns:
        List[Dict[str, Any]]: List of server configurations
    """
    base_url = "https://api.github.com/repos/modelcontextprotocol/servers/contents/src"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        # Get list of files in the src directory
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        server_configs = []
        
        for item in response.json():
            if item["type"] != "dir":  # Skip non-directory items
                continue
                
            # Get the README.md content
            readme_url = f"https://raw.githubusercontent.com/modelcontextprotocol/servers/main/src/{item['name']}/README.md"
            readme_response = requests.get(readme_url)
            
            if readme_response.status_code != 200:
                continue
                
            # Extract JSON configuration from the README
            config = extract_json_from_markdown(readme_response.text)
            
            
            # Iterate over each JSON object in the list
            for json_obj in config:
                if isinstance(json_obj, dict):  # Ensure it's a dictionary
                    for key, value in json_obj.items():
                        if key == "mcpServers" and isinstance(value, dict):
                            server_configs.append(value)
                            break
            
        return server_configs
    
    except requests.RequestException as e:
        print(f"Error fetching MCP servers: {e}")
        return []

def update_mcp_servers_file(output_file: str = None) -> None:
    """
    Updates the MCP servers JSON file with the latest configurations.
    
    Args:
        output_file (str): Path to the output JSON file. Defaults to the root directory of the repository.
    """
    # Determine the root directory of the repository
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    output_file = output_file or os.path.join(root_dir, "mcp_servers.json")
    
    servers = fetch_mcp_servers()
    
    if servers:
        with open(output_file, 'w') as f:
            json.dump(servers, f, indent=2)
        print(f"Successfully updated {output_file} with {len(servers)} server configurations")
    else:
        print("No server configurations were fetched. File not updated.")

if __name__ == "__main__":
    # Update the MCP servers file in the root directory of the repository
    update_mcp_servers_file()