from typing import Final

from mangum import Mangum

from litellm.proxy.proxy_server import app

handler: Final = Mangum(app, lifespan="on")
