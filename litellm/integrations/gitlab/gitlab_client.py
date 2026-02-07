"""
GitLab API client for fetching files from GitLab repositories.
Now supports selecting a tag via `config["tag"]`; falls back to branch ("main").
"""

import base64
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from litellm.llms.custom_httpx.http_handler import HTTPHandler


class GitLabClient:
    """
    Client for interacting with the GitLab API to fetch files.

    Supports:
    - Authentication with personal/access tokens or OAuth bearer tokens
    - Fetching file contents from repositories (raw endpoint with JSON fallback)
    - Namespace/project path or numeric project ID addressing
    - Ref selection via tag (preferred) or branch (default "main")
    - Directory listing via the repository tree API
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GitLab client.

        Args:
            config: Dictionary containing:
                - project: Project path ("group/subgroup/repo") or numeric project ID (str|int) [required]
                - access_token: GitLab personal/access token or OAuth token [required] (str)
                - auth_method: 'token' (default; sends Private-Token) or 'oauth' (Authorization: Bearer)
                - tag: Tag name to fetch from (takes precedence over branch if provided)
                - branch: Branch to fetch from (default: "main")
                - base_url: Base GitLab API URL (default: "https://gitlab.com/api/v4")
        """
        project = config.get("project")
        access_token = config.get("access_token")
        if project is None or access_token is None:
            raise ValueError("project and access_token are required")

        self.project: str | int = project
        self.access_token: str = str(access_token)
        self.auth_method = config.get("auth_method", "token")  # 'token' or 'oauth'
        self.branch = config.get("branch", None)
        if not self.branch:
            self.branch = 'main'
        self.tag = config.get("tag")
        self.base_url = config.get("base_url", "https://gitlab.com/api/v4")

        if not all([self.project, self.access_token]):
            raise ValueError("project and access_token are required")

        # Effective ref: prefer tag if provided, else branch ("main")
        self.ref = str(self.tag or self.branch)

        # Build headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.auth_method == "oauth":
            self.headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            # Default GitLab token header
            self.headers["Private-Token"] = self.access_token

        # Project identifier must be URL-encoded (slashes become %2F)
        self._project_enc = quote(str(self.project), safe="")

        # HTTP handler
        self.http_handler = HTTPHandler()

    # ------------------------
    # Core helpers
    # ------------------------

    def _file_raw_url(self, file_path: str, *, ref: Optional[str] = None) -> str:
        file_enc = quote(file_path, safe="")
        ref_q = quote(ref or self.ref, safe="")
        return f"{self.base_url}/projects/{self._project_enc}/repository/files/{file_enc}/raw?ref={ref_q}"

    def _file_json_url(self, file_path: str, *, ref: Optional[str] = None) -> str:
        file_enc = quote(file_path, safe="")
        ref_q = quote(ref or self.ref, safe="")
        return f"{self.base_url}/projects/{self._project_enc}/repository/files/{file_enc}?ref={ref_q}"

    def _tree_url(self, directory_path: str = "", recursive: bool = False, *, ref: Optional[str] = None) -> str:
        path_q = f"&path={quote(directory_path, safe='')}" if directory_path else ""
        rec_q = "&recursive=true" if recursive else ""
        ref_q = quote(ref or self.ref, safe="")
        return f"{self.base_url}/projects/{self._project_enc}/repository/tree?ref={ref_q}{path_q}{rec_q}"

    # ------------------------
    # Public API
    # ------------------------

    def set_ref(self, ref: str) -> None:
        """Override the default ref (tag/branch) for subsequent calls."""
        if not ref:
            raise ValueError("ref must be a non-empty string")
        self.ref = ref

    def get_file_content(self, file_path: str, *, ref: Optional[str] = None) -> Optional[str]:
        """
        Fetch the content of a file from the GitLab repository at the given ref
        (tag, branch, or commit SHA). If `ref` is None, uses self.ref.

        Strategy:
          1) Try the RAW endpoint (returns bytes of the file)
          2) Fallback to the JSON endpoint (returns base64-encoded content)

        Returns:
            File content as UTF-8 string, or None if file not found.
        """
        raw_url = self._file_raw_url(file_path, ref=ref)

        try:
            resp = self.http_handler.get(raw_url, headers=self.headers)
            if resp.status_code == 404:
                # Fallback to JSON endpoint
                return self._get_file_content_via_json(file_path, ref=ref)
            resp.raise_for_status()

            ctype = (resp.headers.get("content-type") or "").lower()
            if ctype.startswith("text/") or "charset=" in ctype or ctype.startswith("application/json"):
                return resp.text
            try:
                return resp.content.decode("utf-8")
            except Exception:
                return resp.content.decode("utf-8", errors="replace")

        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return None
            if status == 403:
                raise Exception(
                    f"Access denied to file '{file_path}'. Check your GitLab permissions for project '{self.project}'."
                )
            if status == 401:
                raise Exception("Authentication failed. Check your GitLab token and auth_method.")
            raise Exception(f"Failed to fetch file '{file_path}': {e}")

    def _get_file_content_via_json(self, file_path: str, *, ref: Optional[str] = None) -> Optional[str]:
        """
        Fallback for get_file_content(): use the JSON file API which returns base64 content.
        """
        json_url = self._file_json_url(file_path, ref=ref)
        try:
            resp = self.http_handler.get(json_url, headers=self.headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content")
            encoding = data.get("encoding", "")
            if content and encoding == "base64":
                try:
                    return base64.b64decode(content).decode("utf-8")
                except Exception:
                    return base64.b64decode(content).decode("utf-8", errors="replace")
            return content
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return None
            if status == 403:
                raise Exception(
                    f"Access denied to file '{file_path}'. Check your GitLab permissions for project '{self.project}'."
                )
            if status == 401:
                raise Exception("Authentication failed. Check your GitLab token and auth_method.")
            raise Exception(f"Failed to fetch file '{file_path}' via JSON endpoint: {e}")

    def list_files(
            self,
            directory_path: str = "",
            file_extension: str = ".prompt",
            recursive: bool = False,
            *,
            ref: Optional[str] = None,
    ) -> List[str]:
        """
        List files in a directory with a specific extension using the repository tree API.

        Args:
            directory_path: Directory path in the repository (empty for repo root)
            file_extension: File extension to filter by (default: .prompt)
            recursive: If True, traverses subdirectories
            ref: Optional override (tag/branch/SHA). Defaults to self.ref.

        Returns:
            List of file paths (relative to repo root)
        """
        url = self._tree_url(directory_path, recursive=recursive, ref=ref)

        try:
            resp = self.http_handler.get(url, headers=self.headers)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()

            data = resp.json() or []
            files: List[str] = []
            for item in data:
                if item.get("type") == "blob":
                    file_path = item.get("path", "")
                    if not file_extension or file_path.endswith(file_extension):
                        files.append(file_path)
            return files

        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return []
            if status == 403:
                raise Exception(
                    f"Access denied to directory '{directory_path}'. Check your GitLab permissions for project '{self.project}'."
                )
            if status == 401:
                raise Exception("Authentication failed. Check your GitLab token and auth_method.")
            raise Exception(f"Failed to list files in '{directory_path}': {e}")

    def get_repository_info(self) -> Dict[str, Any]:
        """Get information about the project/repository."""
        url = f"{self.base_url}/projects/{self._project_enc}"
        try:
            resp = self.http_handler.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise Exception(f"Failed to get repository info: {e}")

    def test_connection(self) -> bool:
        """Test the connection to the GitLab project."""
        try:
            self.get_repository_info()
            return True
        except Exception:
            return False

    def get_branches(self) -> List[Dict[str, Any]]:
        """Get list of branches in the repository."""
        url = f"{self.base_url}/projects/{self._project_enc}/repository/branches"
        try:
            resp = self.http_handler.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            raise Exception(f"Failed to get branches: {e}")

    def get_file_metadata(self, file_path: str, *, ref: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get minimal metadata about a file via RAW endpoint headers at a given ref.

        Args:
            file_path: Path to the file in the repository.
            ref: Optional override (tag/branch/SHA). Defaults to self.ref.
        """
        url = self._file_raw_url(file_path, ref=ref)
        try:
            headers = dict(self.headers)
            headers["Range"] = "bytes=0-0"
            resp = self.http_handler.get(url, headers=headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return {
                "content_type": resp.headers.get("content-type"),
                "content_length": resp.headers.get("content-length"),
                "last_modified": resp.headers.get("last-modified"),
            }
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return None
            raise Exception(f"Failed to get file metadata for '{file_path}': {e}")

    def close(self):
        """Close the HTTP handler to free resources."""
        if hasattr(self, "http_handler"):
            self.http_handler.close()
