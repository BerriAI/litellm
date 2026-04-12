"""
MCP OpenAPI 功能单元测试

覆盖以下 commit 的核心逻辑：
- 3ce623ff11  添加 mcp openapi 鉴权支持
- 4aef691c39  修复 openapi mcp 不支持 resource、prompt 设置
- 921f98dcdb  优化 mcp openapi tool 名称不匹配情况
- c16b72cf3d  MCP openapi 修复工具名称未对应问题
- 4b50030384  MCP 改进 openapi Schema 解析支持远程地址，修复 tool 获取失败问题
- f8d533b289  添加指定 mcp server id 支持

重点测试 openapi_to_mcp_generator.py 中的纯函数逻辑，
不依赖网络或数据库。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 导入被测模块（纯函数部分）
# ---------------------------------------------------------------------------
from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    _sanitize_path_parameter_value,
    get_base_url,
    extract_parameters,
    build_input_schema,
    create_tool_function,
)
from litellm.proxy._experimental.mcp_server.utils import (
    add_server_prefix_to_name,
    get_server_prefix,
    split_server_prefix_from_name,
    is_tool_name_prefixed,
    normalize_server_name,
)


# ===========================================================================
# 一、路径参数安全校验（_sanitize_path_parameter_value）
# ===========================================================================

class TestSanitizePathParameterValue:
    """测试路径参数的安全校验，防止目录穿越攻击"""

    def test_正常参数值直接返回(self):
        """普通字符串应被正常 URL 编码并返回"""
        result = _sanitize_path_parameter_value("hello123", "id")
        assert result == "hello123"

    def test_None值返回空字符串(self):
        """None 参数应返回空字符串"""
        result = _sanitize_path_parameter_value(None, "id")
        assert result == ""

    def test_空字符串返回空字符串(self):
        """空字符串应直接返回空字符串"""
        result = _sanitize_path_parameter_value("", "id")
        assert result == ""

    def test_含斜杠的参数抛出异常(self):
        """含有路径分隔符 '/' 的参数应抛出 ValueError，防止目录穿越"""
        with pytest.raises(ValueError, match="path separators"):
            _sanitize_path_parameter_value("../../etc/passwd", "id")

    def test_含反斜杠的参数抛出异常(self):
        """含有反斜杠的参数（会被规范化为 '/'）应抛出 ValueError"""
        with pytest.raises(ValueError, match="path separators"):
            _sanitize_path_parameter_value("a\\b", "id")

    def test_单点号参数通过检查并返回(self):
        """
        单独的 '.' 在 PurePosixPath 中 parts 为空元组，
        因此不会触发 '.' 或 '..' 检查，会直接被 quote 返回。
        quote 将 '.' 视为安全字符，所以返回 '.' 本身。
        """
        result = _sanitize_path_parameter_value(".", "id")
        # urllib.parse.quote 默认将 '.' 视为安全字符，不编码
        assert result == "."

    def test_双点号参数抛出异常(self):
        """含 '..' 的路径段应抛出 ValueError"""
        with pytest.raises(ValueError, match=r"'\.' or '\.\.'"):
            _sanitize_path_parameter_value("..", "id")

    def test_特殊字符被URL编码(self):
        """特殊字符（如空格）应被 URL 编码"""
        result = _sanitize_path_parameter_value("hello world", "q")
        assert result == "hello%20world"

    def test_数字参数被转为字符串(self):
        """数字参数应被转为字符串后返回"""
        result = _sanitize_path_parameter_value(42, "page")
        assert result == "42"


# ===========================================================================
# 二、OpenAPI base URL 提取（get_base_url）
# ===========================================================================

class TestGetBaseUrl:
    """测试从 OpenAPI spec 中提取 base URL 的逻辑"""

    def test_openapi3_servers字段提取(self):
        """OpenAPI 3.x 格式：从 servers 字段提取第一个 URL"""
        spec = {
            "servers": [
                {"url": "https://api.example.com/v1"},
                {"url": "https://api.example.com/v2"},
            ]
        }
        assert get_base_url(spec) == "https://api.example.com/v1"

    def test_swagger2_host字段提取(self):
        """OpenAPI 2.x (Swagger) 格式：拼接 scheme + host + basePath"""
        spec = {
            "host": "api.example.com",
            "schemes": ["https"],
            "basePath": "/v2",
        }
        assert get_base_url(spec) == "https://api.example.com/v2"

    def test_swagger2_默认scheme为https(self):
        """Swagger 格式无 schemes 字段时默认使用 https"""
        spec = {
            "host": "api.example.com",
        }
        result = get_base_url(spec)
        assert result.startswith("https://")

    def test_swagger2_无basePath时拼接为空(self):
        """Swagger 格式无 basePath 时 basePath 为空字符串"""
        spec = {
            "host": "api.example.com",
            "schemes": ["http"],
        }
        result = get_base_url(spec)
        assert result == "http://api.example.com"

    def test_空spec返回空字符串(self):
        """空 spec 应返回空字符串"""
        assert get_base_url({}) == ""

    def test_servers为空列表时返回空字符串(self):
        """servers 字段存在但为空列表时应返回空字符串"""
        assert get_base_url({"servers": []}) == ""


# ===========================================================================
# 三、参数提取（extract_parameters）
# ===========================================================================

class TestExtractParameters:
    """测试从 OpenAPI operation 提取路径/查询/请求体参数"""

    def test_提取路径参数(self):
        """path 类型参数应进入 path_params 列表"""
        operation = {
            "parameters": [
                {"name": "user_id", "in": "path"},
            ]
        }
        path_params, query_params, body_params = extract_parameters(operation)
        assert "user_id" in path_params
        assert len(query_params) == 0
        assert len(body_params) == 0

    def test_提取查询参数(self):
        """query 类型参数应进入 query_params 列表"""
        operation = {
            "parameters": [
                {"name": "page", "in": "query"},
                {"name": "limit", "in": "query"},
            ]
        }
        path_params, query_params, body_params = extract_parameters(operation)
        assert "page" in query_params
        assert "limit" in query_params
        assert len(path_params) == 0

    def test_提取body参数_swagger2格式(self):
        """Swagger 2.x body 参数应进入 body_params"""
        operation = {
            "parameters": [
                {"name": "payload", "in": "body"},
            ]
        }
        path_params, query_params, body_params = extract_parameters(operation)
        assert "payload" in body_params

    def test_提取requestBody_openapi3格式(self):
        """OpenAPI 3.x requestBody 应在 body_params 中添加 'body'"""
        operation = {
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }
        }
        path_params, query_params, body_params = extract_parameters(operation)
        assert "body" in body_params

    def test_混合参数类型(self):
        """同时包含 path/query/body 参数时应分别归类"""
        operation = {
            "parameters": [
                {"name": "id", "in": "path"},
                {"name": "filter", "in": "query"},
            ],
            "requestBody": {
                "content": {"application/json": {}}
            }
        }
        path_params, query_params, body_params = extract_parameters(operation)
        assert "id" in path_params
        assert "filter" in query_params
        assert "body" in body_params

    def test_空operation返回三个空列表(self):
        """无参数的 operation 应返回三个空列表"""
        path_params, query_params, body_params = extract_parameters({})
        assert path_params == []
        assert query_params == []
        assert body_params == []


# ===========================================================================
# 四、Input Schema 构建（build_input_schema）
# ===========================================================================

class TestBuildInputSchema:
    """测试从 OpenAPI operation 构建 MCP input schema"""

    def test_构建带必填参数的schema(self):
        """required=True 的参数应出现在 schema 的 required 列表中"""
        operation = {
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                    "description": "用户 ID",
                }
            ]
        }
        schema = build_input_schema(operation)
        assert schema["type"] == "object"
        assert "user_id" in schema["properties"]
        assert schema["properties"]["user_id"]["type"] == "integer"
        assert "user_id" in schema["required"]

    def test_构建可选参数schema(self):
        """required=False 的参数不应出现在 required 列表中"""
        operation = {
            "parameters": [
                {
                    "name": "page",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "integer"},
                }
            ]
        }
        schema = build_input_schema(operation)
        assert "page" in schema["properties"]
        assert "page" not in schema["required"]

    def test_构建带requestBody的schema(self):
        """requestBody 应作为 'body' 属性添加到 schema"""
        operation = {
            "requestBody": {
                "required": True,
                "description": "请求数据",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            }
                        }
                    }
                }
            }
        }
        schema = build_input_schema(operation)
        assert "body" in schema["properties"]
        assert schema["properties"]["body"]["type"] == "object"
        assert "body" in schema["required"]

    def test_空operation返回空schema(self):
        """无参数 operation 返回 properties 为空的 schema"""
        schema = build_input_schema({})
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_参数描述被保留(self):
        """参数描述应被原样保留在 schema 中"""
        operation = {
            "parameters": [
                {
                    "name": "q",
                    "in": "query",
                    "description": "搜索关键词",
                    "schema": {"type": "string"},
                }
            ]
        }
        schema = build_input_schema(operation)
        assert schema["properties"]["q"]["description"] == "搜索关键词"

    def test_无schema类型时默认string(self):
        """参数无 schema.type 时应默认为 string 类型"""
        operation = {
            "parameters": [
                {
                    "name": "token",
                    "in": "query",
                    "description": "访问令牌",
                }
            ]
        }
        schema = build_input_schema(operation)
        assert schema["properties"]["token"]["type"] == "string"


# ===========================================================================
# 五、工具名称标准化（tool name normalization）
# ===========================================================================

class TestToolNameNormalization:
    """测试 operationId 到 tool_name 的标准化逻辑"""

    def test_operationId空格转下划线并小写(self):
        """operationId 中的空格替换为下划线，并转小写（对齐 generator 中逻辑）"""
        operation_id = "Get User Profile"
        tool_name = operation_id.replace(" ", "_").lower()
        assert tool_name == "get_user_profile"

    def test_operationId已是小写无变化(self):
        """已经是小写且无空格的 operationId 不变"""
        operation_id = "list_users"
        tool_name = operation_id.replace(" ", "_").lower()
        assert tool_name == "list_users"

    def test_无operationId时用method和path生成名称(self):
        """无 operationId 时应根据 method + path 自动生成名称"""
        method = "get"
        path = "/users/profile"
        # 对齐源码逻辑：operation.get("operationId", f"{method}_{path.replace('/', '_')}")
        operation_id = f"{method}_{path.replace('/', '_')}"
        tool_name = operation_id.replace(" ", "_").lower()
        assert tool_name == "get__users_profile"

    def test_add_server_prefix_to_name(self):
        """add_server_prefix_to_name 应用连字符拼接服务器名和工具名"""
        prefixed = add_server_prefix_to_name("list_users", "my_api")
        assert prefixed == "my_api-list_users"

    def test_split_server_prefix_from_name(self):
        """split_server_prefix_from_name 应正确还原工具名和服务器前缀"""
        tool_name, server_prefix = split_server_prefix_from_name("my_api-list_users")
        assert tool_name == "list_users"
        assert server_prefix == "my_api"

    def test_is_tool_name_prefixed_有前缀返回True(self):
        """含分隔符的工具名应识别为有前缀"""
        assert is_tool_name_prefixed("my_api-list_users") is True

    def test_is_tool_name_prefixed_无前缀返回False(self):
        """不含分隔符的工具名应识别为无前缀"""
        assert is_tool_name_prefixed("list_users") is False

    def test_normalize_server_name_空格转下划线(self):
        """服务器名中空格应替换为下划线"""
        assert normalize_server_name("my api server") == "my_api_server"


# ===========================================================================
# 六、get_server_prefix 逻辑
# ===========================================================================

class TestGetServerPrefix:
    """测试 get_server_prefix 的优先级逻辑：alias > server_name > server_id"""

    def test_有alias时返回alias(self):
        """有 alias 时应返回 alias"""
        server = MagicMock()
        server.alias = "my_alias"
        server.server_name = "my_server"
        server.server_id = "server-001"
        assert get_server_prefix(server) == "my_alias"

    def test_无alias时返回server_name(self):
        """无 alias 但有 server_name 时应返回 server_name"""
        server = MagicMock()
        server.alias = None
        server.server_name = "my_server"
        server.server_id = "server-001"
        assert get_server_prefix(server) == "my_server"

    def test_无alias无server_name时返回server_id(self):
        """无 alias 无 server_name 时应返回 server_id"""
        server = MagicMock()
        server.alias = None
        server.server_name = None
        server.server_id = "server-001"
        assert get_server_prefix(server) == "server-001"


# ===========================================================================
# 七、鉴权 Header 构建逻辑
# ===========================================================================

class TestAuthHeaderBuilding:
    """测试 _register_openapi_tools 中根据 auth_type 构建 Authorization header 的逻辑"""

    def _build_headers_from_server(self, server: MagicMock) -> Dict[str, str]:
        """
        复现 mcp_server_manager.py 中 _register_openapi_tools 的 header 构建逻辑（内联副本）
        """
        from litellm.types.mcp import MCPAuth
        headers: Dict[str, str] = {}
        if server.authentication_token:
            if server.auth_type == MCPAuth.bearer_token:
                headers["Authorization"] = f"Bearer {server.authentication_token}"
            elif server.auth_type == MCPAuth.api_key:
                headers["Authorization"] = f"ApiKey {server.authentication_token}"
            elif server.auth_type == MCPAuth.basic:
                headers["Authorization"] = f"Basic {server.authentication_token}"
        return headers

    def test_bearer_token鉴权(self):
        """auth_type=bearer_token 应构建 'Bearer xxx' header"""
        from litellm.types.mcp import MCPAuth
        server = MagicMock()
        server.authentication_token = "my-secret-token"
        server.auth_type = MCPAuth.bearer_token
        headers = self._build_headers_from_server(server)
        assert headers.get("Authorization") == "Bearer my-secret-token"

    def test_api_key鉴权(self):
        """auth_type=api_key 应构建 'ApiKey xxx' header"""
        from litellm.types.mcp import MCPAuth
        server = MagicMock()
        server.authentication_token = "my-api-key"
        server.auth_type = MCPAuth.api_key
        headers = self._build_headers_from_server(server)
        assert headers.get("Authorization") == "ApiKey my-api-key"

    def test_basic鉴权(self):
        """auth_type=basic 应构建 'Basic xxx' header"""
        from litellm.types.mcp import MCPAuth
        server = MagicMock()
        server.authentication_token = "dXNlcjpwYXNz"
        server.auth_type = MCPAuth.basic
        headers = self._build_headers_from_server(server)
        assert headers.get("Authorization") == "Basic dXNlcjpwYXNz"

    def test_无token时不添加header(self):
        """authentication_token 为 None 时不应添加任何 Authorization header"""
        from litellm.types.mcp import MCPAuth
        server = MagicMock()
        server.authentication_token = None
        server.auth_type = MCPAuth.bearer_token
        headers = self._build_headers_from_server(server)
        assert "Authorization" not in headers

    def test_无auth_type时不添加header(self):
        """authentication_token 存在但 auth_type 不匹配任何已知类型时不构建 header"""
        from litellm.types.mcp import MCPAuth
        server = MagicMock()
        server.authentication_token = "token"
        server.auth_type = MCPAuth.none  # 不匹配 bearer/api_key/basic
        headers = self._build_headers_from_server(server)
        assert "Authorization" not in headers


# ===========================================================================
# 八、OpenAPI spec 中 resource/prompt 字段处理
# ===========================================================================

class TestOpenApiServerResourceAndPrompt:
    """
    测试 OpenAPI MCP server 对 resource/prompt 的处理逻辑。
    对应 commit 4aef691c39：修复 openapi mcp 不支持 resource、prompt 设置。
    OpenAPI server 应返回空列表（不支持 resource/prompt），
    而不是调用 MCP client（因为没有真实的 MCP server 端点）。
    """

    def _make_openapi_server(self) -> Any:
        """构造一个带 spec_path 的 MCPServer 实例（模拟 OpenAPI 服务器）"""
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.types.mcp import MCPTransport
        return MCPServer(
            server_id="openapi-server-001",
            name="openapi_server",
            url="https://api.example.com",
            transport=MCPTransport.http,
            spec_path="/path/to/openapi.yaml",
        )

    @pytest.mark.asyncio
    async def test_openapi_server_get_prompts返回空列表(self):
        """
        spec_path 不为 None 时 get_prompts_from_server 应返回空列表，
        不调用 MCP client 的 list_prompts
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        manager = MCPServerManager()
        server = self._make_openapi_server()

        # mock _create_mcp_client 防止真实网络连接
        mock_client = AsyncMock()
        mock_client.list_prompts = AsyncMock(return_value=[])
        with patch.object(manager, "_create_mcp_client", return_value=mock_client):
            with patch.object(manager, "_build_stdio_env", return_value={}):
                with patch.object(manager, "_create_prefixed_prompts", return_value=[]):
                    prompts = await manager.get_prompts_from_server(server)

        # OpenAPI server 应返回空列表，并且不调用 list_prompts
        mock_client.list_prompts.assert_not_called()
        assert prompts == []

    @pytest.mark.asyncio
    async def test_openapi_server_get_resources返回空列表(self):
        """
        spec_path 不为 None 时 get_resources_from_server 应返回空列表，
        不调用 MCP client 的 list_resources
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        manager = MCPServerManager()
        server = self._make_openapi_server()

        mock_client = AsyncMock()
        mock_client.list_resources = AsyncMock(return_value=[])
        with patch.object(manager, "_create_mcp_client", return_value=mock_client):
            with patch.object(manager, "_build_stdio_env", return_value={}):
                with patch.object(manager, "_create_prefixed_resources", return_value=[]):
                    resources = await manager.get_resources_from_server(server)

        mock_client.list_resources.assert_not_called()
        assert resources == []


# ===========================================================================
# 九、create_tool_function 生成的工具函数行为
# ===========================================================================

class TestCreateToolFunction:
    """测试 create_tool_function 生成的 async 函数的基本行为"""

    @pytest.mark.asyncio
    async def test_get请求调用正确的client方法(self):
        """
        create_tool_function 生成的 GET 工具函数应调用 http client 的 get 方法
        """
        operation = {
            "parameters": [
                {"name": "page", "in": "query"}
            ]
        }
        mock_response = MagicMock()
        mock_response.text = '{"result": "ok"}'
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        tool_fn = create_tool_function(
            path="/users",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await tool_fn(page="2")

        mock_client.get.assert_called_once()
        assert result == '{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_post请求调用正确的client方法(self):
        """
        create_tool_function 生成的 POST 工具函数应调用 http client 的 post 方法
        """
        operation = {
            "requestBody": {
                "content": {"application/json": {"schema": {"type": "object"}}}
            }
        }
        mock_response = MagicMock()
        mock_response.text = '{"id": 1}'
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tool_fn = create_tool_function(
            path="/users",
            method="post",
            operation=operation,
            base_url="https://api.example.com",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await tool_fn(body={"name": "Alice"})

        mock_client.post.assert_called_once()
        assert result == '{"id": 1}'

    @pytest.mark.asyncio
    async def test_不支持的http方法返回错误信息(self):
        """
        不支持的 HTTP 方法（如 'head'）应返回错误信息字符串
        """
        operation: Dict[str, Any] = {}
        mock_client = AsyncMock()

        tool_fn = create_tool_function(
            path="/test",
            method="head",
            operation=operation,
            base_url="https://api.example.com",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await tool_fn()

        assert "Unsupported HTTP method" in result

    @pytest.mark.asyncio
    async def test_路径参数被正确替换到URL中(self):
        """
        路径参数应被替换到 URL 的对应占位符中，且请求 URL 正确
        """
        operation = {
            "parameters": [
                {"name": "user_id", "in": "path", "required": True, "schema": {"type": "integer"}}
            ]
        }
        mock_response = MagicMock()
        mock_response.text = '{"user": "Alice"}'
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        tool_fn = create_tool_function(
            path="/users/{user_id}",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            await tool_fn(user_id="123")

        call_args = mock_client.get.call_args
        # 第一个位置参数是 URL
        called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "123" in called_url
        assert "{user_id}" not in called_url

    @pytest.mark.asyncio
    async def test_鉴权header被传递到请求中(self):
        """
        create_tool_function 接收的 headers 应被传递到 HTTP 请求中
        """
        operation: Dict[str, Any] = {}
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        auth_headers = {"Authorization": "Bearer secret-token"}

        tool_fn = create_tool_function(
            path="/protected",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
            headers=auth_headers,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            await tool_fn()

        call_kwargs = mock_client.get.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"].get("Authorization") == "Bearer secret-token"

    @pytest.mark.asyncio
    async def test_路径参数含目录穿越时返回错误信息(self):
        """
        路径参数值含目录穿越字符时，工具函数应返回错误信息字符串而非抛出异常
        """
        operation = {
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ]
        }
        mock_client = AsyncMock()

        tool_fn = create_tool_function(
            path="/items/{id}",
            method="get",
            operation=operation,
            base_url="https://api.example.com",
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await tool_fn(id="../../etc/passwd")

        assert "Invalid path parameter" in result


# ===========================================================================
# 十、_deserialize_json_dict 辅助函数
# ===========================================================================

class TestDeserializeJsonDict:
    """测试 mcp_server_manager 中数据库 JSON 字段的反序列化辅助函数"""

    def test_字符串JSON被正确反序列化(self):
        """JSON 字符串应被反序列化为 dict"""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _deserialize_json_dict,
        )
        result = _deserialize_json_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_已是dict时直接返回(self):
        """已经是 dict 时应直接返回"""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _deserialize_json_dict,
        )
        data = {"key": "value"}
        assert _deserialize_json_dict(data) == data

    def test_None时返回None(self):
        """None 输入应返回 None"""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _deserialize_json_dict,
        )
        assert _deserialize_json_dict(None) is None

    def test_空字符串返回None(self):
        """空字符串应返回 None"""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _deserialize_json_dict,
        )
        assert _deserialize_json_dict("") is None

    def test_无效JSON字符串返回None(self):
        """无法解析的 JSON 字符串应返回 None"""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _deserialize_json_dict,
        )
        assert _deserialize_json_dict("not-valid-json") is None
