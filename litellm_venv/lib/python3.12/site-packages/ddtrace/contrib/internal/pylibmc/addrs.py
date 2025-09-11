translate_server_specs = None

try:
    # NOTE: we rely on an undocumented method to parse addresses,
    # so be a bit defensive and don't assume it exists.
    from pylibmc.client import translate_server_specs
except ImportError:
    pass


def parse_addresses(addrs):
    if not translate_server_specs:
        return []
    return translate_server_specs(addrs)
