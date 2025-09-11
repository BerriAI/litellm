from twisted.internet import reactor
from twisted.web.wsgi import WSGIResource

from .. import exposition, REGISTRY

MetricsResource = lambda registry=REGISTRY: WSGIResource(
    reactor, reactor.getThreadPool(), exposition.make_wsgi_app(registry)
)
