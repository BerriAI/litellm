import os, asyncio, pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

@pytest.mark.ndsmoke
def test_chutes_chat_live_optional():
    if not (os.getenv('CHUTES_API_KEY') or os.getenv('CHUTES_API_TOKEN')):
        pytest.skip('CHUTES_API_KEY not set; skipping live chutes ndsmoke')
    
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import arouter_call
    messages=[{"role":"user","content":"Say hello in one short sentence."}]
    
    out = asyncio.run(
        arouter_call(model='chutes/deepseek-ai/DeepSeek-R1', messages=messages)
    )
    text = (out.get('choices',[{}])[0].get('message',{}) or {}).get('content','')
    assert isinstance(text, str) and len(text.strip())>0
