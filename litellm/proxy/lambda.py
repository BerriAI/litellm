from mangum import Mangum
from litellm.proxy.proxy_server import app

handler = Mangum(app, lifespan="on")
