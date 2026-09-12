import json,os,subprocess
from pathlib import Path
root=Path('/Users/jvalluru/LiteLLM-work/litellm-lit-7078')
out=Path('/verification')
base=(out/'before-sha').read_text().strip()
env={**os.environ,'PATH':'/opt/lit6634-venv/bin:'+os.environ['PATH'],'UV_PROJECT_ENVIRONMENT':'/opt/lit6634-venv'}
checks={
 'lock':['uv','lock','--check'],
 'format':['ruff','format','--check','litellm/proxy/_experimental/mcp_server/discoverable_endpoints.py'],
 'ruff':['ruff','check','litellm'],
 'test-tree':['ruff','check','--config','ruff-tests.toml','tests'],
 'strict':['python','scripts/ruff_strict_gate.py','--base',base],
 'type-discipline':['python','scripts/type_discipline_gate.py','--base',base],
 'test-quality':['python','scripts/test_quality_gate.py','--base',base],
 'basedpyright':['python','scripts/type_check_gate.py','--base',base],
}
results={}
for name,command in checks.items():
 print('Starting '+name,flush=True)
 with (out/(name+'-gate.log')).open('w') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 results[name]=result.returncode
 (out/'gate-results.json').write_text(json.dumps(results,indent=2))
 print(name+': '+str(result.returncode),flush=True)
