# FORK_PATCHES.md - fork 补丁清单

本仓库是 [BerriAI/litellm](https://github.com/BerriAI/litellm) 的内部 fork（origin = yzp0111/litellm.git）。
本清单登记 fork 对上游的每一处本地改动，目标是"最小分叉"：所有改动可枚举、可回归、升级上游后可 replay。

## 基线 SHA

| 字段 | 值 |
| --- | --- |
| 基线 SHA（fork HEAD） | `1a6642ee2edd9587eeab008864d7e9bba4cc9c26` |
| 基线分支 | `litellm_internal_staging` |
| pyproject 版本 | `1.95.0`（pyproject.toml:3） |
| 对应上游分支 | `upstream/litellm_internal_staging`（本 fork 跟踪上游 staging 分支，而非 main） |
| 基线校验时间 | 2026-07-29（fetch upstream 后校验） |

基线校验结论：基线 SHA `1a6642ee2e` 是 `upstream/litellm_internal_staging` 历史中的提交
（`git merge-base --is-ancestor HEAD upstream/litellm_internal_staging` 退出码 0），
且 fork 侧本地改动为 0 处（`git log upstream/litellm_internal_staging..HEAD` 为空）。
校验时上游 staging 已前进到 `c274cf321c5c35c629220a89bb497d15b56f870f`（领先基线 15 个提交）。

### 关于 v1.95.0 release tag 的说明

pyproject 版本是 1.95.0，但上游**尚不存在 `v1.95.0` release tag**（2026-07-29 fetch 时核实）。
上游 1.95.0 周期的 tag 仅有 main 线上的预发布 tag：

- `v1.95.0-dev.2` = `7495b1f282268be78b606770b43e9d7e407eff9f`（最接近的 1.95.0 周期 tag）
- `v1.95.0-dev.1` = `f2479cc704f6e63d5510929d30ce8e11ffe43467`
- 上游最新正式 release tag 为 `v1.94.0`

因此基线锚点是上游 staging 分支的祖先关系（见上），而非某个 release tag。
注意：`v1.95.0-dev.*` 在 main 线上，与本 fork 跟踪的 staging 线不同源，
`git diff upstream/tags/v1.95.0-dev.2..HEAD --stat` 的巨大 diff（约 1561 个文件）
反映的是 main 线与 staging 线的分支差异，不是 fork 的本地改动。
`git describe HEAD` = `v1.93.0-dev.3-1184-g1a6642ee2e`。

参考命令：

```bash
git rev-parse upstream/tags/v1.95.0-dev.2   # 7495b1f282268be78b606770b43e9d7e407eff9f
git merge-base --is-ancestor HEAD upstream/litellm_internal_staging && echo OK
git log upstream/litellm_internal_staging..HEAD --oneline | wc -l   # fork 本地提交数，基线时为 0
git diff upstream/litellm_internal_staging..HEAD --stat             # fork 本地 diff，基线时为空
git merge-base HEAD upstream/main   # 7cd009caf7467f5839ab4acc1ae9b6e58160cee9
```

## 补丁登记表（对上游文件的改动）

每处对上游文件的本地改动都必须登记：文件、原因、对应回归测试、基线 SHA。
升级上游 replay 时，以回归测试红绿判定补丁是否存活，而不是以 diff 能否 apply 判定。

| # | 文件 | 改动 | 原因 | 回归测试 | 基线 SHA |
| --- | --- | --- | --- | --- | --- |
| - | （暂无）建立基线时 fork 对上游文件改动为 0 处 | - | - | - | `1a6642ee2edd9587eeab008864d7e9bba4cc9c26` |

## fork 自有文件（非上游改动）

以下文件为 fork 新增、上游不存在，不存在冲突问题，但需登记以便审计。

| 文件 | 用途 | 引入 commit |
| --- | --- | --- |
| `FORK_PATCHES.md` | 本清单 | 见 `git log -- FORK_PATCHES.md` |

## 分支与提交约定

- 分支命名统一用 `feature/litellm_<desc>` 形式。
  该形式同时满足两条约束：pre-push 钩子要求 `<type>/<desc>` 带斜杠
  （type 允许值：feature、bugfix、hotfix、release、chore，见 `.githooks/pre-push`），
  以及 CLAUDE.md 要求分支带 `litellm_` 前缀。
  非 feature 类工作用对应 type，例如 `chore/litellm_<desc>`。
- base 分支统一为 `litellm_internal_staging`，不对 main 提 PR，不切默认分支到 main。
- 钩子保护分支（可直接 push，不受命名约束）：`main`、`litellm_internal_staging`、`dependabot/*`、`gh-readonly-queue/*`。
- 提交信息遵循 Conventional Commits（commit-msg 钩子强制，描述首字母小写），
  允许类型：feat、fix、docs、style、refactor、perf、test、build、ci、chore、revert。
- 不加 `Co-Authored-By`，不加任何 AI 署名（CLAUDE.md 约定）。
- 每个 PR 至少 1 个测试（硬性要求，CONTRIBUTING.md）。
- 提交前跑 `make format` 与 `make pre-commit`。

## git 钩子

`make install-hooks` 通过 `git config core.hooksPath .githooks` 启用钩子
（`.githooks/commit-msg` 与 `.githooks/pre-push`）。
本 clone 同时将两个钩子镜像复制到了 `.git/hooks/` 下，作为 hooksPath 被 unset 时的兜底。
每个新 clone 需重跑一次 `make install-hooks`；紧急情况下可用 `--no-verify` 绕过。

## 升级 replay 要点（详见后续运维 Runbook）

1. `git fetch upstream --tags`
2. 以 `upstream/litellm_internal_staging` 为目标做 merge/rebase
3. 冲突解决后跑本清单登记的回归测试：测试绿则补丁存活，测试红则补丁需重建
