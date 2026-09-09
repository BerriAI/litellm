
import json, sys
mode, failure_method = sys.argv[1:]
for line in sys.stdin:
    request = json.loads(line)
    if "method" not in request or "id" not in request:
        continue
    if request["method"] == failure_method:
        if mode == "bad-json":
            print("secret-invalid-json", flush=True)
            continue
        if mode == "closed":
            sys.exit(0)
        if mode == "silent":
            print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Waiting"}}), flush=True)
            continue
    if request["method"] == "initialize":
        result = {"protocolVersion": "2099-01-01" if mode == "unknown" else request["params"]["protocolVersion"], "capabilities": {"tools": {}, "logging": {}}, "serverInfo": {"name": "diagnostic", "version": "1"}}
    elif request["method"] == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Listing tools"}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "unmatched", "result": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"}), flush=True)
        result = {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
    else:
        result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
