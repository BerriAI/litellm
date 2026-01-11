import requests
import yaml
from pathlib import Path

def fetch_openrouter_models():
    """Fetch all models from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['data']

def create_model_entry(model):
    """Convert OpenRouter model to config format"""
    model_id = model['id']
    model_name = model['name']
    
    entry = {
        'model_name': f"{model_name}",
        'litellm_params': {
            'model': f"openrouter/{model_id}",
            'api_key': 'os.environ/OPENROUTER_API_KEY'
        },
        'model_info': {
            'id': model_id,
            'description': f"{model['name']} from OpenRouter. {model['description']}",
            'max_tokens': model['context_length']
       }
    }
    
    # Add pricing if available - keep as native number type
    if 'pricing' in model:
        if 'prompt' in model['pricing']:
            entry['litellm_params']['input_cost_per_token'] = str(model['pricing']['prompt'])
        if 'completion' in model['pricing']:
            entry['litellm_params']['output_cost_per_token'] = str(model['pricing']['completion'])
    
    # Add context length if available
    #if 'context_length' in model:
        #entry['litellm_params']['max_tokens'] = model['context_length']
    
    return entry

def generate_config():
    """Generate complete config.yaml"""
    config = {
        'general_settings': {
            'user_header_mappings': [
                {
                    'header_name': 'X-OpenWebUI-User-Id',
                    'litellm_user_role': 'customer'
                },
                {
                    'header_name': 'X-OpenWebUI-User-Email',
                    'litellm_user_role': 'internal_user'
                }
            ],
            'store_model_in_db': True,
            'store_prompts_in_spend_logs': True,
            'maximum_spend_logs_retention_period': '7d'
        },
        'litellm_settings': {
            'callbacks': ['smtp_email', 'langfuse', 'openmeter']
        },
        'model_list': []
    }
    
    # Fetch and process models
    models = fetch_openrouter_models()
    config['model_list'] = [create_model_entry(model) for model in models]
    
    return config

def save_config(config):
    """Save config to YAML file"""

    output_path = Path(__file__).parent / 'proxy_config.yaml'
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to {output_path}")
    print(f"Total models: {len(config['model_list'])}")

if __name__ == '__main__':
    config = generate_config()
    save_config(config)
