# project
from ddtrace._trace.utils_redis import _extract_conn_tags as extract_redis_tags
from ddtrace.contrib.internal.pylibmc.addrs import parse_addresses
from ddtrace.ext import net


def _resource_from_cache_prefix(resource, cache):
    """
    Combine the resource name with the cache prefix (if any)
    """
    if getattr(cache, "key_prefix", None):
        name = "{} {}".format(resource, cache.key_prefix)
    else:
        name = resource

    # enforce lowercase to make the output nicer to read
    return name.lower()


def _extract_client(cache):
    """
    Get the client from the cache instance according to the current operation
    """
    client = getattr(cache, "_client", None)
    if client is None:
        # flask-caching has _read_clients & _write_client for the redis backend
        # These use the same connection so just try to get a reference to one of them.
        # flask-caching < 2.0.0 uses _read_clients so look for that one too.
        for attr in ("_write_client", "_read_client", "_read_clients"):
            client = getattr(cache, attr, None)
            if client is not None:
                break
    return client


def _extract_conn_tags(client):
    """
    For the given client extracts connection tags
    """
    tags = {}

    if hasattr(client, "servers"):
        # Memcached backend supports an address pool
        if isinstance(client.servers, list) and len(client.servers) > 0:
            # use the first address of the pool as a host because
            # the code doesn't expose more information
            contact_point = client.servers[0].address
            tags[net.TARGET_HOST] = contact_point[0]
            tags[net.TARGET_PORT] = contact_point[1]
            tags[net.SERVER_ADDRESS] = contact_point[0]
    elif hasattr(client, "connection_pool"):
        # Redis main connection
        redis_tags = extract_redis_tags(client.connection_pool.connection_kwargs)
        tags.update(**redis_tags)
    elif hasattr(client, "addresses"):
        # pylibmc
        # FIXME[matt] should we memoize this?
        addrs = parse_addresses(client.addresses)
        if addrs:
            _, host, port, _ = addrs[0]
            tags[net.TARGET_PORT] = port
            tags[net.TARGET_HOST] = host
            tags[net.SERVER_ADDRESS] = host
    return tags
