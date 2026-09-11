import ast
import json
import re
import subprocess
from pathlib import Path
root = Path('/Users/jvalluru/LiteLLM-work/lit-7498-verification')
coverage = json.loads((root / 'coverage.json').read_text())
base = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/litellm_internal_staging'], text=True).strip()
changed = {}
current = None
for line in subprocess.check_output(['git', 'diff', '--unified=0', base, '--', 'litellm'], text=True).splitlines():
    if line.startswith('+++ b/'):
        current = line[6:]
        changed[current] = set()
    elif line.startswith('@@'):
        match = re.search(r'\+(\d+)(?:,(\d+))?', line)
        start, size = int(match[1]), int(match[2] or 1)
        changed[current].update(range(start, start + size))
results = []
for name, lines in changed.items():
    if name not in coverage['files']:
        continue
    data = coverage['files'][name]
    executed, missing = set(data['executed_lines']), set(data['missing_lines'])
    touched = []
    for node in ast.walk(ast.parse(Path(name).read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and lines.intersection(range(node.lineno, node.end_lineno + 1)):
            covered_arcs = [arc for arc in data['executed_branches'] if node.lineno <= arc[0] <= node.end_lineno]
            missing_arcs = [arc for arc in data['missing_branches'] if node.lineno <= arc[0] <= node.end_lineno]
            touched.append({'function': node.name, 'covered_branches': len(covered_arcs), 'missing_branches': missing_arcs})
    results.append({'file': name, 'changed_executable_lines': sorted(lines & (executed | missing)), 'uncovered_changed_lines': sorted(lines & missing), 'touched_functions': touched})
result = {'tip': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'base': base, 'files': results}
(root/'branch-coverage.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
