import os, socket, asyncio, pytest, base64

# 1x1 PNG red dot (tiny) base64
_RED_DOT = (
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhIQEBAQEA8QEA8QEA8QDw8QEA8QFREWFhURFRUYHSggGBolHRUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGi0fHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJ8BPgMBIgACEQEDEQH/xAAUAAEAAAAAAAAAAAAAAAAAAAAF/8QAFhEBAQEAAAAAAAAAAAAAAAAAAQIR/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCDgA//2Q=='
)


def _ollama_host_port():
    host_env = os.getenv('OLLAMA_HOST')
    if host_env and ':' in host_env:
        h, p = host_env.rsplit(':', 1)
        try:
            return h, int(p)
        except Exception:
            pass
    return os.getenv('OLLAMA_HOST', '127.0.0.1'), int(os.getenv('OLLAMA_PORT', '11434'))


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
def test_ollama_multimodal_live_optional():
    host, port = _ollama_host_port()
    if not _can_connect(host, port):
        pytest.skip(f'ollama not reachable on {host}:{port}')

    async def run():
        from litellm.router import Router
        import httpx
        # Prefer explicit env, else require gemma3:12b by default
        try:
            tags = httpx.get(f'http://{host}:{port}/api/tags', timeout=1.0).json()
            names = { m.get('name') for m in (tags.get('models') or []) }
            env_model = os.getenv('LITELLM_DEFAULT_VISION_MODEL','')
            if env_model.startswith('ollama/'):
                # If user provided explicit vision model, verify it's present
                candidate = env_model.split('ollama/',1)[1]
                if candidate in names:
                    model = env_model
                else:
                    pytest.skip(f"ollama model '{candidate}' not available; set LITELLM_DEFAULT_VISION_MODEL correctly or pull it")
            else:
                # Default: gemma3:12b
                if 'gemma3:12b' in names:
                    model = 'ollama/gemma3:12b'
                else:
                    pytest.skip('ollama gemma3:12b not available; skipping')
        except Exception:
            pytest.skip('ollama /api/tags not available')
        r = Router(model_list=[{"model_name":"m","litellm_params":{"model": model}}])
        # Fetch remote image and embed as data URL
        import httpx
        img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/320px-Grosser_Panda.JPG'
        img_bytes = httpx.get(img_url, timeout=5.0).content
        data_url = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode('ascii')

        messages = [{
            'role':'user',
            'content': [
                { 'type':'text', 'text':'Describe this image in one short phrase.' },
                { 'type':'image_url', 'image_url': { 'url': data_url }},
            ]
        }]
        resp = await r.acompletion(model='m', messages=messages, timeout=20)
        try:
            text = getattr(getattr(resp.choices[0],'message',{}),'content',None)
        except Exception:
            text = resp.get('choices',[{}])[0].get('message',{}).get('content')
        assert isinstance(text, str) and len(text) > 0

    asyncio.run(run())
