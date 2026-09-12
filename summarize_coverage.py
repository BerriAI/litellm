import json, re, subprocess, ast
from pathlib import Path
p=Path(__file__).resolve().parent
root=Path('/Users/jvalluru/LiteLLM-work/litellm-lit-7078')
base=(p/'before-sha').read_text().strip()
c=json.loads((p/'coverage.json').read_text())
out={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'merge_base':base,'files':{}}
for f,r in c['files'].items():
    diff=subprocess.check_output(['git','diff','--unified=0',base,'HEAD','--',f],cwd=root,text=True)
    changed=set()
    for line in diff.splitlines():
        if line.startswith('@@'):
            m=re.search(r'\+(\d+)(?:,(\d+))?',line)
            start=int(m[1]); count=int(m[2] or 1); changed.update(range(start,start+count))
    added=changed & (set(r['executed_lines'])|set(r['missing_lines']))
    missing=added & set(r['missing_lines'])
    nodes = [node for node in ast.walk(ast.parse((root/f).read_text())) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    touched = {node.name for node in nodes if any(node.lineno <= line <= node.end_lineno for line in changed)}
    funcs = {name: data for name, data in r.get('functions', {}).items() if name in touched}
    out['files'][f]={'changed_executable_lines':sorted(added),'uncovered_changed_lines':sorted(missing),'changed_line_coverage':100*(len(added)-len(missing))/len(added) if added else None,'touched_functions':funcs}
(p/'coverage-summary.json').write_text(json.dumps(out,indent=2))
print('Changed executable lines:',sum(len(r['changed_executable_lines']) for r in out['files'].values()))
print('Uncovered changed executable lines:',sum(len(r['uncovered_changed_lines']) for r in out['files'].values()))
