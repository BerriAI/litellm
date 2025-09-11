class SpanTypes(object):
    CACHE = "cache"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    GRPC = "grpc"
    GRAPHQL = "graphql"
    HTTP = "http"
    MONGODB = "mongodb"
    REDIS = "redis"
    SERVERLESS = "serverless"
    SQL = "sql"
    TEMPLATE = "template"
    TEST = "test"
    WEB = "web"
    WORKER = "worker"
    AUTH = "auth"
    SYSTEM = "system"
    LLM = "llm"


class SpanKind(object):
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


EXIT_SPAN_TYPES = frozenset(
    {
        SpanTypes.CACHE,
        SpanTypes.CASSANDRA,
        SpanTypes.ELASTICSEARCH,
        SpanTypes.GRPC,
        SpanTypes.HTTP,
        SpanTypes.REDIS,
        SpanTypes.SQL,
        SpanTypes.WORKER,
    }
)
