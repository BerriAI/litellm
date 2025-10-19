"""
GitLab prompt manager with configurable prompts folder.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from jinja2 import DictLoader, Environment, select_autoescape

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.integrations.gitlab.gitlab_client import GitLabClient


GITLAB_PREFIX = "gitlab::"

def encode_prompt_id(raw_id: str) -> str:
    """Convert GitLab path IDs like 'invoice/extract' → 'gitlab::invoice::extract'"""
    if raw_id.startswith(GITLAB_PREFIX):
        return raw_id  # already encoded
    return f"{GITLAB_PREFIX}{raw_id.replace('/', '::')}"

def decode_prompt_id(encoded_id: str) -> str:
    """Convert 'gitlab::invoice::extract' → 'invoice/extract'"""
    if not encoded_id.startswith(GITLAB_PREFIX):
        return encoded_id
    return encoded_id[len(GITLAB_PREFIX):].replace("::", "/")


class GitLabPromptTemplate:
    def __init__(
            self,
            template_id: str,
            content: str,
            metadata: Dict[str, Any],
            model: Optional[str] = None,
    ):
        self.template_id = template_id
        self.content = content
        self.metadata = metadata
        self.model = model or metadata.get("model")
        self.temperature = metadata.get("temperature")
        self.max_tokens = metadata.get("max_tokens")
        self.input_schema = metadata.get("input", {}).get("schema", {})
        self.optional_params = {
            k: v for k, v in metadata.items() if k not in ["model", "input", "content"]
        }

    def __repr__(self):
        return f"GitLabPromptTemplate(id='{self.template_id}', model='{self.model}')"


class GitLabTemplateManager:
    """
    Manager for loading and rendering .prompt files from GitLab repositories.

    New: supports `prompts_path` (or `folder`) in gitlab_config to scope where prompts live.
    """


    def __init__(
            self,
            gitlab_config: Dict[str, Any],
            prompt_id: Optional[str] = None,
            ref: Optional[str] = None,
            gitlab_client: Optional[GitLabClient] = None
    ):
        self.gitlab_config = dict(gitlab_config)
        self.prompt_id = prompt_id
        self.prompts: Dict[str, GitLabPromptTemplate] = {}
        self.gitlab_client = gitlab_client or GitLabClient(self.gitlab_config)

        if ref:
            self.gitlab_client.set_ref(ref)

        # Folder inside repo to look for prompts (e.g., "prompts" or "prompts/chat")
        self.prompts_path: str = (
                self.gitlab_config.get("prompts_path")
                or self.gitlab_config.get("folder")
                or ""
        ).strip("/")

        self.jinja_env = Environment(
            loader=DictLoader({}),
            autoescape=select_autoescape(["html", "xml"]),
            variable_start_string="{{",
            variable_end_string="}}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
        )

        if self.prompt_id:
            self._load_prompt_from_gitlab(self.prompt_id)

    # ---------- path helpers ----------

    def _id_to_repo_path(self, prompt_id: str) -> str:
        """Map a prompt_id to a repo path (respects prompts_path and adds .prompt)."""
        prompt_id = decode_prompt_id(prompt_id)
        if self.prompts_path:
            return f"{self.prompts_path}/{prompt_id}.prompt"
        return f"{prompt_id}.prompt"

    def _repo_path_to_id(self, repo_path: str) -> str:
        """
        Map a repo path like 'prompts/chat/greeting.prompt' to an ID relative
        to prompts_path without the extension (e.g., 'chat/greeting').
        """
        path = repo_path.strip("/")
        if self.prompts_path and path.startswith(self.prompts_path.strip("/") + "/"):
            path = path[len(self.prompts_path.strip("/")) + 1 :]
        if path.endswith(".prompt"):
            path = path[: -len(".prompt")]
        return encode_prompt_id(path)

    # ---------- loading ----------

    def _load_prompt_from_gitlab(self, prompt_id: str, *, ref: Optional[str] = None) -> None:
        """Load a specific .prompt file from GitLab (scoped under prompts_path if set)."""
        try:
            # prompt_id = decode_prompt_id(prompt_id)
            file_path = self._id_to_repo_path(prompt_id)
            prompt_content = self.gitlab_client.get_file_content(file_path, ref=ref)
            if prompt_content:
                template = self._parse_prompt_file(prompt_content, prompt_id)
                self.prompts[prompt_id] = template
        except Exception as e:
            raise Exception(f"Failed to load prompt '{encode_prompt_id(prompt_id)}' from GitLab: {e}")

    def load_all_prompts(self, *, recursive: bool = True) -> List[str]:
        """
        Eagerly load all .prompt files from prompts_path. Returns loaded IDs.
        """
        files = self.list_templates(recursive=recursive)
        loaded: List[str] = []
        for pid in files:
            if pid not in self.prompts:
                self._load_prompt_from_gitlab(pid)
            loaded.append(pid)
        return loaded

    # ---------- parsing & rendering ----------

    def _parse_prompt_file(
            self, content: str, prompt_id: str
    ) -> GitLabPromptTemplate:
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1].strip()
                template_content = parts[2].strip()
            else:
                frontmatter_str = ""
                template_content = content
        else:
            frontmatter_str = ""
            template_content = content

        metadata: Dict[str, Any] = {}
        if frontmatter_str:
            try:
                import yaml
                metadata = yaml.safe_load(frontmatter_str) or {}
            except ImportError:
                metadata = self._parse_yaml_basic(frontmatter_str)
            except Exception:
                metadata = {}

        return GitLabPromptTemplate(
            template_id=prompt_id,
            content=template_content,
            metadata=metadata,
        )

    def _parse_yaml_basic(self, yaml_str: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for line in yaml_str.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.lower() in ["true", "false"]:
                    result[key] = value.lower() == "true"
                elif value.isdigit():
                    result[key] = int(value)
                elif value.replace(".", "").isdigit():
                    try:
                        result[key] = float(value)
                    except Exception:
                        result[key] = value
                else:
                    result[key] = value.strip("\"'")
        return result

    def render_template(
            self, template_id: str, variables: Optional[Dict[str, Any]] = None
    ) -> str:
        if template_id not in self.prompts:
            raise ValueError(f"Template '{template_id}' not found")
        template = self.prompts[template_id]
        jinja_template = self.jinja_env.from_string(template.content)
        return jinja_template.render(**(variables or {}))

    def get_template(self, template_id: str) -> Optional[GitLabPromptTemplate]:
        return self.prompts.get(template_id)

    def list_templates(self, *, recursive: bool = True) -> List[str]:
        """
        List available prompt IDs under prompts_path (no extension).
        Compatible with both list_files signatures:
        - list_files(directory_path=..., file_extension=..., recursive=...)
        - list_files(path=..., ref=None, recursive=...)
        """
        # First try the "new" signature (directory_path/file_extension)
        try:
            files = self.gitlab_client.list_files(
                directory_path=self.prompts_path,
                file_extension=".prompt",
                recursive=recursive,
            )
            base = self.prompts_path.strip("/")
            out: List[str] = []
            for p in files or []:
                path = str(p).strip("/")
                if base and not path.startswith(base + "/"):
                    # if the client returns extra files outside the folder, skip them
                    continue
                if not path.endswith(".prompt"):
                    continue
                out.append(self._repo_path_to_id(path))
            return out
        except TypeError:
            # Fallback to the "classic" signature
            raw = self.gitlab_client.list_files(
                directory_path=self.prompts_path or "",
                ref=None,
                recursive=recursive,
            )
            # Classic returns GitLab tree entries; filter *.prompt blobs
            files = []
            for f in (raw or []):
                if isinstance(f, dict) and f.get("type") == "blob" and str(f.get("path", "")).endswith(".prompt") and 'path' in f:
                    files.append(f['path'])

            return [self._repo_path_to_id(p) for p in files]


class GitLabPromptManager(CustomPromptManagement):
    """
    GitLab prompt manager with folder support.

    Example config:
        gitlab_config = {
            "project": "group/subgroup/repo",
            "access_token": "glpat_***",
            "tag": "v1.2.3",          # optional; takes precedence
            "branch": "main",         # default fallback
            "prompts_path": "prompts/chat"
        }
    """

    def __init__(
            self,
            gitlab_config: Dict[str, Any],
            prompt_id: Optional[str] = None,
            ref: Optional[str] = None,  # tag/branch/SHA override
            gitlab_client: Optional[GitLabClient] = None
    ):
        self.gitlab_config = gitlab_config
        self.prompt_id = prompt_id
        self._prompt_manager: Optional[GitLabTemplateManager] = None
        self._ref_override = ref
        self._injected_gitlab_client = gitlab_client
        if self.prompt_id:
            self._prompt_manager = GitLabTemplateManager(
                gitlab_config=self.gitlab_config,
                prompt_id=self.prompt_id,
                ref=self._ref_override,
            )

    @property
    def integration_name(self) -> str:
        return "gitlab"

    @property
    def prompt_manager(self) -> GitLabTemplateManager:
        if self._prompt_manager is None:
            self._prompt_manager = GitLabTemplateManager(
                gitlab_config=self.gitlab_config,
                prompt_id=self.prompt_id,
                ref=self._ref_override,
                gitlab_client=self._injected_gitlab_client
            )
        return self._prompt_manager

    def get_prompt_template(
            self,
            prompt_id: str,
            prompt_variables: Optional[Dict[str, Any]] = None,
            *,
            ref: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if prompt_id not in self.prompt_manager.prompts:
            self.prompt_manager._load_prompt_from_gitlab(prompt_id, ref=ref)

        template = self.prompt_manager.get_template(prompt_id)
        if not template:
            raise ValueError(f"Prompt template '{prompt_id}' not found")

        rendered_prompt = self.prompt_manager.render_template(
            prompt_id, prompt_variables or {}
        )

        metadata = {
            "model": template.model,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
            **template.optional_params,
        }
        return rendered_prompt, metadata

    def pre_call_hook(
            self,
            user_id: Optional[str],
            messages: List[AllMessageValues],
            function_call: Optional[Union[Dict[str, Any], str]] = None,
            litellm_params: Optional[Dict[str, Any]] = None,
            prompt_id: Optional[str] = None,
            prompt_variables: Optional[Dict[str, Any]] = None,
            prompt_version: Optional[str] = None,
            **kwargs,
    ) -> Tuple[List[AllMessageValues], Optional[Dict[str, Any]]]:
        if not prompt_id:
            return messages, litellm_params
        try:
            # Precedence: explicit prompt_version → per-call git_ref kwarg → manager override → config default
            git_ref = prompt_version or kwargs.get("git_ref") or self._ref_override

            rendered_prompt, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables, ref=git_ref
            )
            parsed_messages = self._parse_prompt_to_messages(rendered_prompt)

            if parsed_messages:
                final_messages: List[AllMessageValues] = parsed_messages
            else:
                final_messages = [{"role": "user", "content": rendered_prompt}] + messages  # type: ignore

            if litellm_params is None:
                litellm_params = {}

            if prompt_metadata.get("model"):
                litellm_params["model"] = prompt_metadata["model"]

            for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                if param in prompt_metadata:
                    litellm_params[param] = prompt_metadata[param]

            return final_messages, litellm_params
        except Exception as e:
            import litellm
            litellm._logging.verbose_proxy_logger.error(f"Error in GitLab prompt pre_call_hook: {e}")
            return messages, litellm_params


    def _parse_prompt_to_messages(self, prompt_content: str) -> List[AllMessageValues]:
        messages: List[AllMessageValues] = []
        lines = prompt_content.strip().split("\n")
        current_role: Optional[str] = None
        current_content: List[str] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("system:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})  # type: ignore
                current_role = "system"
                current_content = [line[7:].strip()]
            elif low.startswith("user:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})  # type: ignore
                current_role = "user"
                current_content = [line[5:].strip()]
            elif low.startswith("assistant:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})  # type: ignore
                current_role = "assistant"
                current_content = [line[10:].strip()]
            else:
                current_content.append(line)

        if current_role and current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})  # type: ignore
        if not messages and prompt_content.strip():
            messages = [{"role": "user", "content": prompt_content.strip()}]  # type: ignore
        return messages

    def post_call_hook(
            self,
            user_id: Optional[str],
            response: Any,
            input_messages: List[AllMessageValues],
            function_call: Optional[Union[Dict[str, Any], str]] = None,
            litellm_params: Optional[Dict[str, Any]] = None,
            prompt_id: Optional[str] = None,
            prompt_variables: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> Any:
        return response

    def get_available_prompts(self) -> List[str]:
        """
        Return prompt IDs. Prefer already-loaded templates in memory to avoid
        unnecessary network calls (and to make tests deterministic).
        """
        ids = set(self.prompt_manager.prompts.keys())
        try:
            ids.update(self.prompt_manager.list_templates())
        except Exception:
            # If GitLab list fails (auth, network), still return what we've loaded.
            pass
        return sorted(ids)

    def reload_prompts(self) -> None:
        if self.prompt_id:
            self._prompt_manager = None
            _ = self.prompt_manager  # trigger re-init/load

    def should_run_prompt_management(
            self,
            prompt_id: str,
            dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        return True

    def _compile_prompt_helper(
            self,
            prompt_id: str,
            prompt_variables: Optional[dict],
            dynamic_callback_params: StandardCallbackDynamicParams,
            prompt_label: Optional[str] = None,
            prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        try:
            decoded_id = decode_prompt_id(prompt_id)
            if decoded_id not in self.prompt_manager.prompts:
                git_ref = getattr(dynamic_callback_params, "extra", {}).get("git_ref") if hasattr(dynamic_callback_params, "extra") else None
                self.prompt_manager._load_prompt_from_gitlab(decoded_id, ref=git_ref)


            rendered_prompt, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables
            )

            messages = self._parse_prompt_to_messages(rendered_prompt)
            template_model = prompt_metadata.get("model")

            optional_params: Dict[str, Any] = {}
            for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                if param in prompt_metadata:
                    optional_params[param] = prompt_metadata[param]

            return PromptManagementClient(
                prompt_id=prompt_id,
                prompt_template=messages,
                prompt_template_model=template_model,
                prompt_template_optional_params=optional_params,
                completed_messages=None,
            )
        except Exception as e:
            raise ValueError(f"Error compiling prompt '{prompt_id}': {e}")

    def get_chat_completion_prompt(
            self,
            model: str,
            messages: List[AllMessageValues],
            non_default_params: dict,
            prompt_id: Optional[str],
            prompt_variables: Optional[dict],
            dynamic_callback_params: StandardCallbackDynamicParams,
            prompt_label: Optional[str] = None,
            prompt_version: Optional[int] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        return PromptManagementBase.get_chat_completion_prompt(
            self,
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
            prompt_label,
            prompt_version,
        )


class GitLabPromptCache:
    """
    Cache all .prompt files from a GitLab repo into memory.

    - Keys are the *repo file paths* (e.g. "prompts/chat/greet/hi.prompt")
      mapped to JSON-like dicts containing content + metadata.
    - Also exposes a by-ID view (ID == path relative to prompts_path without ".prompt",
      e.g. "greet/hi").

    Usage:

        cfg = {
            "project": "group/subgroup/repo",
            "access_token": "glpat_***",
            "prompts_path": "prompts/chat",  # optional, can be empty for repo root
            # "branch": "main",              # default is "main"
            # "tag": "v1.2.3",               # takes precedence over branch
            # "base_url": "https://gitlab.com/api/v4"  # default
        }

        cache = GitLabPromptCache(cfg)
        cache.load_all()  # fetch + parse all .prompt files

        print(cache.list_files())  # repo file paths
        print(cache.list_ids())    # template IDs relative to prompts_path

        prompt_json = cache.get_by_file("prompts/chat/greet/hi.prompt")
        prompt_json2 = cache.get_by_id("greet/hi")

        # If GitLab content changes and you want to refresh:
        cache.reload()  # re-scan and refresh all
    """

    def __init__(
            self,
            gitlab_config: Dict[str, Any],
            *,
            ref: Optional[str] = None,
            gitlab_client: Optional[GitLabClient] = None,
    ) -> None:
        # Build a PromptManager (which internally builds TemplateManager + Client)
        self.prompt_manager = GitLabPromptManager(
            gitlab_config=gitlab_config,
            prompt_id=None,
            ref=ref,
            gitlab_client=gitlab_client,
        )
        self.template_manager: GitLabTemplateManager = self.prompt_manager.prompt_manager

        # In-memory stores
        self._by_file: Dict[str, Dict[str, Any]] = {}
        self._by_id: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Public API
    # -------------------------

    def load_all(self, *, recursive: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Scan GitLab for all .prompt files under prompts_path, load and parse each,
        and return the mapping of repo file path -> JSON-like dict.
        """
        ids = self.template_manager.list_templates(recursive=recursive)  # IDs relative to prompts_path
        for pid in ids:
            # Ensure template is loaded into TemplateManager
            if pid not in self.template_manager.prompts:
                self.template_manager._load_prompt_from_gitlab(pid)

            tmpl = self.template_manager.get_template(pid)
            if tmpl is None:
                # If something raced/failed, try once more
                self.template_manager._load_prompt_from_gitlab(pid)
                tmpl = self.template_manager.get_template(pid)
            if tmpl is None:
                continue

            file_path = self.template_manager._id_to_repo_path(pid)  # "prompts/chat/..../file.prompt"
            entry = self._template_to_json(pid, tmpl)

            self._by_file[file_path] = entry
            # prefixed_id = pid if pid.startswith("gitlab::") else f"gitlab::{pid}"
            encoded_id = encode_prompt_id(pid)
            self._by_id[encoded_id] = entry
            # self._by_id[pid] = entry

        return self._by_id

    def reload(self, *, recursive: bool = True) -> Dict[str, Dict[str, Any]]:
        """Clear the cache and re-load from GitLab."""
        self._by_file.clear()
        self._by_id.clear()
        return self.load_all(recursive=recursive)

    def list_files(self) -> List[str]:
        """Return the repo file paths currently cached."""
        return list(self._by_file.keys())

    def list_ids(self) -> List[str]:
        """Return the template IDs (relative to prompts_path, without extension) currently cached."""
        return list(self._by_id.keys())

    def get_by_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a cached prompt JSON by repo file path."""
        return self._by_file.get(file_path)

    def get_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a cached prompt JSON by prompt ID (relative to prompts_path)."""
        if prompt_id in self._by_id:
            return self._by_id[prompt_id]

        # Try normalized forms
        decoded = decode_prompt_id(prompt_id)
        encoded = encode_prompt_id(decoded)

        return self._by_id.get(encoded) or self._by_id.get(decoded)

    # -------------------------
    # Internals
    # -------------------------

    def _template_to_json(self, prompt_id: str, tmpl: GitLabPromptTemplate) -> Dict[str, Any]:
        """
        Normalize a GitLabPromptTemplate into a JSON-like dict that is easy to serialize.
        """
        # Safer copy of metadata (avoid accidental mutation)
        md = dict(tmpl.metadata or {})

        # Pull standard fields (also present in metadata sometimes)
        model = tmpl.model
        temperature = tmpl.temperature
        max_tokens = tmpl.max_tokens
        optional_params = dict(tmpl.optional_params or {})

        return {
            "id": prompt_id,                                       # e.g. "greet/hi"
            "path": self.template_manager._id_to_repo_path(prompt_id),          # e.g. "prompts/chat/greet/hi.prompt"
            "content": tmpl.content,                                # rendered content (without frontmatter)
            "metadata": md,                                         # parsed frontmatter
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "optional_params": optional_params,
        }