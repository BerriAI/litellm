# LiteLLM 中文前端

中文词库、术语规则、人工修订和构建脚本随 Fork 一起维护。构建时在临时源码副本中应用译文，生成中文前端镜像 `litellm-local:zh-cn`

`litellm_zh_cn` 分支基于 `v1.106.2` 稳定版维护中文功能，`main` 分支用于同步官方上游。`origin` 指向自己的 Fork，`upstream` 指向 `BerriAI/litellm`

## 构建和启动

在仓库根目录操作。本机需要 Docker、Bash 和 tar，Node 与 pnpm 在构建容器内运行

首次部署时，将 `local-dev/.env.example` 复制为根目录 `.env`，填写管理员凭据、数据库密码和内网 IP。将 `local-dev/config.example.yaml` 复制为 `local-dev/config.yaml`。已有部署继续使用原配置

前端源码和后端基础镜像需使用同一版本。当前分支的构建命令为

```bash
./localization/zh-CN/build.sh "$PWD" ghcr.io/berriai/litellm:v1.106.2
./local-dev/start.sh
```

管理界面为 `http://127.0.0.1:4000/ui/`，内网地址由 `.env` 的 `LITELLM_LAN_IP` 决定。启动脚本复用根目录 `.env`、`local-dev/config.yaml` 和原有 PostgreSQL 数据卷

代理需要使用容器可访问的地址，例如

```bash
HTTPS_PROXY=http://host.docker.internal:7897 \
HTTP_PROXY=http://host.docker.internal:7897 \
./localization/zh-CN/build.sh "$PWD" ghcr.io/berriai/litellm:v1.106.2
```

构建执行补丁测试、TypeScript 检查和 Next.js 静态页面生成。构建结束后自动删除临时源码与依赖

## 维护译文

`zh-CN.json` 保存完整词库，`overrides.json` 保存人工修订。`context-overrides.json` 按源文件区分同词异义，例如产品名 Linear 与图表的线性刻度

`terminology.json` 保留 Token、Prompt、Embedding、Completion、Reasoning、Agent 等术语。服务商名称、模型 ID、API 字段、表单取值、URL、代码示例和用户实际数据保持原值。后端动态内容与远程文章沿用上游内容

词库初稿使用离线翻译模型生成，常用页面和已发现误译经过人工修订。低频功能说明可继续在 `overrides.json` 中校对

构建生成 `translation-report.json`，其中 `missing` 列出上游新增或改写的文案及源文件位置。将译文添加到 `overrides.json` 后重新构建。报告属于本地生成文件，由 Git 忽略

## 同步上游

保持 `main` 为官方主分支的镜像

```bash
git fetch upstream main --tags
git switch main
git merge --ff-only upstream/main
git push origin main
git switch litellm_zh_cn
```

中文分支更新稳定版本时，先获取并合并目标稳定版标签，再选择同版本后端镜像重新构建。下面的 `vX.Y.Z` 表示所选稳定版标签

```bash
git fetch upstream tag vX.Y.Z
git merge --no-edit vX.Y.Z
./localization/zh-CN/build.sh "$PWD" ghcr.io/berriai/litellm:vX.Y.Z
```

检查翻译报告、构建结果和页面后，提交中文分支并启动新镜像

```bash
git push origin litellm_zh_cn
./local-dev/start.sh
```

浅克隆需要补齐历史后才能可靠计算合并关系，可先执行 `git fetch --unshallow upstream`。新版本改变根布局或构建工具时，按实际错误调整 `prepare-ui.mjs` 或构建脚本

## 许可证

本目录适用仓库根目录的 [MIT 许可证](../../LICENSE)，保留上游版权与许可声明。`enterprise/` 目录适用其自身许可证，见 [企业版许可证](../../enterprise/LICENSE.md)。离线翻译模型的来源说明见 [NOTICE.md](NOTICE.md)
