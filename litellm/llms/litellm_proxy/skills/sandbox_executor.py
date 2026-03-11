"""
Sandbox Executor for LiteLLM Skills

Executes skill code in a sandboxed environment using llm-sandbox.
Supports Docker, Podman, and Kubernetes backends.
"""

import base64
import os
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger


class SkillsSandboxExecutor:
    """
    Executes skill code in llm-sandbox Docker container.
    
    Responsibilities:
    - Create sandbox session with skill files
    - Install requirements
    - Execute model-generated code
    - Collect generated files (GIFs, images, etc.)
    """

    def __init__(
        self,
        timeout: int = 60,
        backend: str = "docker",
        image: Optional[str] = None,
    ):
        """
        Initialize the sandbox executor.
        
        Args:
            timeout: Maximum execution time in seconds
            backend: Sandbox backend ("docker", "podman", "kubernetes")
            image: Custom Docker image (default: uses llm-sandbox default)
        """
        self.timeout = timeout
        self.backend = backend
        self.image = image
        self._session = None

    def execute(
        self,
        code: str,
        skill_files: Dict[str, bytes],
        requirements: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code with skill files in sandbox.
        
        Args:
            code: Python code to execute
            skill_files: Dict mapping file paths to binary content
            requirements: Optional requirements.txt content
            
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str (if failed),
                "files": [{"name": str, "content_base64": str, "mime_type": str}]
            }
        """
        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            verbose_logger.error(
                "SkillsSandboxExecutor: llm-sandbox not installed. "
                "Install with: pip install llm-sandbox"
            )
            return {
                "success": False,
                "output": "",
                "error": "llm-sandbox not installed. Install with: pip install llm-sandbox",
                "files": [],
            }

        try:
            # Create sandbox session
            session_kwargs: Dict[str, Any] = {
                "lang": "python",
                "verbose": False,
            }
            
            if self.image:
                session_kwargs["image"] = self.image
            
            with SandboxSession(**session_kwargs) as session:
                # 1. Copy skill files into sandbox using copy_to_runtime
                import tempfile

                # Create a temp directory to stage files
                with tempfile.TemporaryDirectory() as tmpdir:
                    for path, content in skill_files.items():
                        # Create the file in temp directory
                        local_path = os.path.join(tmpdir, path)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        with open(local_path, "wb") as f:
                            f.write(content)
                        
                        # Copy to sandbox
                        sandbox_path = f"/sandbox/{path}"
                        session.copy_to_runtime(local_path, sandbox_path)
                
                verbose_logger.debug(
                    f"SkillsSandboxExecutor: Copied {len(skill_files)} files to sandbox"
                )
                
                # 2. Install requirements if present
                req_packages = None
                if requirements:
                    req_packages = requirements.strip().replace("\n", " ")
                elif "requirements.txt" in skill_files:
                    req_content = skill_files["requirements.txt"].decode("utf-8")
                    req_packages = req_content.strip().replace("\n", " ")
                
                if req_packages:
                    # Run pip install as code
                    pip_code = f"""
import subprocess
subprocess.run(['pip', 'install'] + '{req_packages}'.split(), check=True)
"""
                    result = session.run(pip_code)
                    verbose_logger.debug(
                        "SkillsSandboxExecutor: Installed requirements"
                    )
                
                # 3. Execute the code
                # Wrap code to run from /sandbox directory
                wrapped_code = f"""
import os
os.chdir('/sandbox')
import sys
sys.path.insert(0, '/sandbox')

{code}
"""
                result = session.run(wrapped_code)
                
                success = result.exit_code == 0
                output = result.stdout or ""
                error = result.stderr or ""
                
                if success:
                    verbose_logger.debug(
                        "SkillsSandboxExecutor: Code execution succeeded"
                    )
                else:
                    verbose_logger.debug(
                        f"SkillsSandboxExecutor: Code execution failed with exit code {result.exit_code}"
                    )
                    verbose_logger.debug(
                        f"SkillsSandboxExecutor: stderr: {error[:500] if error else 'No stderr'}"
                    )
                    verbose_logger.debug(
                        f"SkillsSandboxExecutor: stdout: {output[:500] if output else 'No stdout'}"
                    )
                
                # 4. Collect generated files
                generated_files = self._collect_generated_files(session, skill_files)
                
                return {
                    "success": success,
                    "output": output,
                    "error": error,
                    "files": generated_files,
                }
                
        except Exception as e:
            verbose_logger.error(
                f"SkillsSandboxExecutor: Execution failed: {e}"
            )
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "files": [],
            }

    def _collect_generated_files(
        self,
        session: Any,
        original_files: Dict[str, bytes],
    ) -> List[Dict[str, Any]]:
        """
        Collect files generated during execution.
        
        Looks for new files in /sandbox that weren't in the original skill files.
        Focuses on common output types: GIF, PNG, JPG, PDF, CSV, etc.
        
        Args:
            session: The sandbox session
            original_files: Original skill files (to exclude)
            
        Returns:
            List of generated files with base64 content
        """
        generated_files: List[Dict[str, Any]] = []
        
        try:
            import tempfile

            # List files in /sandbox using Python code
            list_code = """
import os
import json
files = []
for root, dirs, filenames in os.walk('/sandbox'):
    for f in filenames:
        if f.endswith(('.gif', '.png', '.jpg', '.jpeg', '.pdf', '.csv', '.json')):
            files.append(os.path.join(root, f))
print(json.dumps(files))
"""
            result = session.run(list_code)
            
            if result.exit_code == 0 and result.stdout:
                import json
                try:
                    filepaths = json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    filepaths = []
                
                for filepath in filepaths:
                    if not filepath:
                        continue
                    
                    # Get relative path
                    rel_path = filepath.replace("/sandbox/", "")
                    
                    # Skip if it was an original file
                    if rel_path in original_files:
                        continue
                    
                    # Copy file from sandbox using copy_from_runtime
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    try:
                        session.copy_from_runtime(filepath, tmp_path)
                        
                        with open(tmp_path, "rb") as f:
                            content = f.read()
                        
                        content_b64 = base64.b64encode(content).decode("utf-8")
                        generated_files.append({
                            "name": os.path.basename(filepath),
                            "path": rel_path,
                            "content_base64": content_b64,
                            "mime_type": self._get_mime_type(filepath),
                        })
                        
                        verbose_logger.debug(
                            f"SkillsSandboxExecutor: Collected generated file: {rel_path}"
                        )
                    except Exception as e:
                        verbose_logger.warning(
                            f"SkillsSandboxExecutor: Error copying file {filepath}: {e}"
                        )
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                            
        except Exception as e:
            verbose_logger.warning(
                f"SkillsSandboxExecutor: Error collecting generated files: {e}"
            )
        
        return generated_files

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for a file based on extension."""
        ext = filename.lower().split(".")[-1]
        return {
            "gif": "image/gif",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "pdf": "application/pdf",
            "csv": "text/csv",
            "json": "application/json",
            "txt": "text/plain",
        }.get(ext, "application/octet-stream")

