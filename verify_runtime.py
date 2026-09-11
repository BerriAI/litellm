import subprocess,sys,json,hashlib
from pathlib import Path
root=Path(__file__).parent;sha,label=sys.argv[1:3]
repo=Path(__import__('os').environ['LITELLM_SOURCE'])
files=['litellm/proxy/_experimental/mcp_server/mcp_server_manager.py','litellm/proxy/_experimental/mcp_server/openapi_to_mcp_generator.py','litellm/llms/custom_httpx/http_handler.py']
expected={f:hashlib.sha256(subprocess.check_output(['git','show',sha+':'+f],cwd=repo)).hexdigest() for f in files}
program='from pathlib import Path;import hashlib,json;files='+repr(files)+';print(json.dumps({f:hashlib.sha256(Path("/app/.venv/lib/python3.13/site-packages").joinpath(f).read_bytes()).hexdigest() for f in files}))'
tag='litellm-4896:'+sha[:10]
actual=json.loads(subprocess.check_output(['docker','run','--rm','--cpus','0.25','--memory','128m','--entrypoint','python',tag,'-c',program],text=True))
assert actual==expected,'Runtime source does not match commit'
image_id=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
data={'commit':sha,'image_id':image_id,'source_sha256':actual,'matches_git':True}
(root/(label+'-runtime.json')).write_text(json.dumps(data,indent=2));print(json.dumps(data))
