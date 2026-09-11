import ast,json,re,subprocess
from pathlib import Path
root=Path(__file__).parent
repo=Path(__import__('os').environ['LITELLM_SOURCE'])
base=subprocess.check_output(['git','merge-base','HEAD','origin/litellm_internal_staging'],cwd=repo,text=True).strip()
tip=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
report=json.loads((root/'coverage.json').read_text())
files={}
for file in ['litellm/proxy/_experimental/mcp_server/mcp_server_manager.py','litellm/proxy/_experimental/mcp_server/openapi_to_mcp_generator.py','litellm/llms/custom_httpx/http_handler.py']:
 diff=subprocess.check_output(['git','diff','--unified=0',base,'--',file],cwd=repo,text=True)
 changed=set()
 for line in diff.splitlines():
  match=re.match(r'@@ .* \+(\d+)(?:,(\d+))? @@',line)
  if match:changed.update(range(int(match[1]),int(match[1])+int(match[2] or 1)))
 cov=report['files'][file]
 executed=set(cov['executed_lines']);missing=set(cov['missing_lines']);executable=changed&(executed|missing)
 functions={}
 wanted={'_openapi_spec_health','health_check_server','load_openapi_spec_async','_OpenAPIHealthProbe.__init__','_OpenAPIHealthProbe.check','AsyncHTTPHandler.get','AsyncHTTPHandler._get_with_response_limit','AsyncHTTPHandler._read_with_response_limit'}
 def visit(nodes,prefix=''):
  for node in nodes:
   if isinstance(node,ast.ClassDef):visit(node.body,prefix+node.name+'.')
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and (prefix+node.name in wanted or node.name in wanted):
    lines=set(range(node.lineno,node.end_lineno+1))
    hit=[b for b in cov['executed_branches'] if b[0] in lines];missed=[b for b in cov['missing_branches'] if b[0] in lines]
    functions[prefix+node.name]={'covered_lines':len(lines&executed),'executable_lines':len(lines&(executed|missing)),'missing_lines':sorted(lines&missing),'covered_branch_arcs':len(hit),'branch_arcs':len(hit)+len(missed),'missing_branch_arcs':missed}
 visit(ast.parse((repo/file).read_text()).body)
 files[file]={'changed_executable_lines':len(executable),'covered_changed_lines':len(executable&executed),'uncovered_changed_lines':sorted(executable&missing),'touched_functions':functions}
count=sum(f['changed_executable_lines'] for f in files.values());covered=sum(f['covered_changed_lines'] for f in files.values())
result={'base':base,'tip':tip,'changed_executable_lines':count,'covered_changed_lines':covered,'changed_line_percent':100*covered/count,'files':files,'note':'Local affected-test coverage only; not repository-wide coverage'}
(root/'changed-coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
