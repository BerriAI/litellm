# Anthropic Messages SDK 对齐

本地回放同一份上游响应，分别执行 Python 和源码编译的 Rust 扩展，比较 HTTP 请求和 SDK 返回值。当前覆盖异步非流式文本、可选参数、400 参数错误及 429 限流错误，暂不覆盖同步、流式和回调

通过公开 SDK `litellm.anthropic_messages` 发起请求，不直接调用底层变换函数。错误用例同时检查兼容层保留上游错误正文和调用者传入的完整模型名，不将原生异常的 `(状态码, 正文)` 元组格式泄露到公开错误消息

先按仓库的锁文件安装依赖并编译扩展，然后运行

```bash
uv sync --frozen
LITELLM_LOCAL_MODEL_COST_MAP=True uv run python -m tests.rust-python-harness run e2e_parity --surface sdk --function messages
```

用例只请求本机回放服务，使用测试凭据，不调用真实模型。缺少编译扩展或候选路径回退到 Python 时，运行失败而非跳过

响应比较保留具体类型和全部公开字段，只排除 `_hidden_params` 内部元数据。成功响应另行检查 `x-litellm-rust` 标记。错误比较使用共享错误模型，包括异常类型、消息、状态码、模型及提供商

用例与防回退校验的测试命令

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/strategies/e2e_parity/sdk/messages/test_case.py -q
```

`test_python_fallback_cannot_satisfy_messages_parity` 故意将候选路由改为 Python，验证校验会拒绝这种假通过。该策略遵守框架的仅本地运行约束，不新增 CI 工作流
