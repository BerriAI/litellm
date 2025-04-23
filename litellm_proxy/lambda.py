from mangum import Mangum
from litellm_proxy.proxy_server import app

handler = Mangum(app, lifespan="on")
