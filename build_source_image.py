import subprocess,sys,tarfile,io
from pathlib import Path
root=Path(__file__).parent
repo=Path(__import__('os').environ['LITELLM_SOURCE'])
sha=sys.argv[1]
source=root/('source-image-'+sha[:10]);source.mkdir(exist_ok=True)
archive=subprocess.check_output(['git','archive',sha,'litellm'],cwd=repo)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(source,filter='data')
(source/'Dockerfile').write_text('FROM litellm-4896:8c82c325ac\nLABEL org.opencontainers.image.revision="'+sha+'"\nCOPY litellm/ /app/.venv/lib/python3.13/site-packages/litellm/\nRUN find /app/.venv/lib/python3.13/site-packages/litellm -type d -name __pycache__ -exec rm -rf {} +\n')
subprocess.run(['docker','build','-t','litellm-4896:'+sha[:10],str(source)],check=True)
