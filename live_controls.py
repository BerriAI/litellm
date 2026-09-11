from api import request,root
import json,subprocess,sys
servers=json.loads((root/'health-servers.json').read_text())
ui_key=(root/'.ui-auth').read_text().removeprefix('Bearer ')
token=subprocess.check_output(['gh','auth','token'],text=True).strip()
results=[]
for alias in ['remote_none','remote_bearer','local_none','local_bearer','native_http']:
 sid=servers[alias]['server_id']
 for case in (['none'] if alias=='native_http' else ['missing','invalid','valid']):
  path='/v1/mcp/server/'+sid+'/user-credential'
  if case=='missing': request('DELETE',path,key=ui_key)
  elif case!='none': request('POST',path,{'credential':token if case=='valid' else 'invalid-lit4896','save':True},key=ui_key)
  ls,lb=request('GET','/mcp-rest/tools/list?server_id='+sid,key=ui_key)
  name='read_wiki_structure' if alias=='native_http' else 'getauthenticateduser'
  cs,cb=request('POST','/mcp-rest/tools/call',{'server_id':sid,'name':name,'arguments':{'repoName':'BerriAI/litellm'} if alias=='native_http' else {}},key=ui_key)
  r={'server':alias,'credential':case,'list_http_status':ls,'tool_count':len(lb.get('tools',[])),'call_http_status':cs,'call_isError':cb.get('isError'),'content_returned':bool(cb.get('content'))}
  if cs!=200 or cb.get('isError'): r['error']=cb
  results.append(r);print(json.dumps(r),flush=True)
(root/(sys.argv[1]+'-controls.json')).write_text(json.dumps(results,indent=2))
