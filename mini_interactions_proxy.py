#!/usr/bin/env python3
"""
Mini proxy server for testing Google SDK with Interactions API.
Routes requests through the litellm.interactions bridge.
"""
import os
import sys
import json

# Add workspace to path
sys.path.insert(0, '/workspace')

# Set API key from environment
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
MASTER_KEY = os.environ.get('LITELLM_MASTER_KEY', 'sk-1234')

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()


@app.post('/v1beta/interactions')
@app.post('/interactions')
async def create_interaction(
    request: Request,
    authorization: str = Header(None),
    x_goog_api_key: str = Header(None),
):
    """Handle interactions API requests."""
    from litellm import interactions
    
    # Check auth
    api_key = None
    if authorization:
        api_key = authorization.replace('Bearer ', '')
    elif x_goog_api_key:
        api_key = x_goog_api_key
    
    if api_key != MASTER_KEY:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    
    # Parse request
    data = await request.json()
    print(f"[PROXY] Received request: {json.dumps(data, indent=2)}")
    
    # Extract parameters - Google SDK sends config.input
    model = data.get('model', 'anthropic/claude-3-5-haiku-20241022')
    
    # Handle Google SDK format where input is in config
    config = data.get('config', {})
    input_data = data.get('input') or config.get('input')
    stream = data.get('stream', False) or config.get('stream', False)
    system_instruction = data.get('system_instruction') or config.get('system_instruction')
    tools = data.get('tools') or config.get('tools')
    generation_config = data.get('generation_config') or config.get('generation_config', {})
    
    print(f"[PROXY] Model: {model}, Input: {input_data}, Stream: {stream}")
    
    try:
        # Build kwargs
        kwargs = {
            'model': model,
            'input': input_data,
            'api_key': ANTHROPIC_API_KEY,
        }
        
        if system_instruction:
            kwargs['system_instruction'] = system_instruction
        if tools:
            kwargs['tools'] = tools
        if generation_config:
            if 'temperature' in generation_config:
                kwargs['temperature'] = generation_config['temperature']
            if 'max_output_tokens' in generation_config:
                kwargs['max_output_tokens'] = generation_config['max_output_tokens']
        
        if stream:
            kwargs['stream'] = True
            response = interactions.create(**kwargs)
            
            async def generate():
                for chunk in response:
                    chunk_data = chunk.model_dump() if hasattr(chunk, 'model_dump') else dict(chunk)
                    yield f'data: {json.dumps(chunk_data)}\n\n'
                yield 'data: [DONE]\n\n'
            
            return StreamingResponse(generate(), media_type='text/event-stream')
        else:
            response = interactions.create(**kwargs)
            response_data = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
            print(f"[PROXY] Response: {json.dumps(response_data, indent=2)}")
            return JSONResponse(response_data)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/')
async def root():
    return {'message': 'Mini Interactions Proxy', 'endpoints': ['/interactions', '/v1beta/interactions']}


if __name__ == '__main__':
    print(f"Starting mini proxy server...")
    print(f"ANTHROPIC_API_KEY: {'*' * 10}...{ANTHROPIC_API_KEY[-4:] if ANTHROPIC_API_KEY else 'NOT SET'}")
    print(f"MASTER_KEY: {MASTER_KEY}")
    uvicorn.run(app, host='0.0.0.0', port=4000, log_level='info')
