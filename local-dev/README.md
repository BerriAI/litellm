# 本地运行

在仓库根目录运行 `./local-dev/start.sh`，通过 Docker Compose 启动中文前端镜像 `litellm-local:zh-cn` 和 PostgreSQL。中文补丁位于仓库内的 `localization/zh-CN`，更新和重建方法见该目录的 `README.md`

运行环境全部位于容器内，本机仅需 Docker 和 Compose

管理界面为 http://127.0.0.1:4000/ui/login/，用户名和密码分别保存在根目录 `.env` 的 `UI_USERNAME` 和 `UI_PASSWORD`

同一内网的设备可使用 `http://<LITELLM_LAN_IP>:4000/ui/login/` 访问，`LITELLM_LAN_IP` 保存在 `.env` 中，Mac 的内网地址改变后需更新该值并重新运行启动脚本

修改 `.env` 中的 `UI_PASSWORD` 并保存后，重新运行启动脚本使新密码生效

`POSTGRES_PASSWORD` 是数据库账号密码，PostgreSQL 已有数据卷时，修改这个变量还需要同步数据库角色的密码

可在管理界面添加 API 提供商、地址、密钥和模型

Prometheus 指标地址为 http://127.0.0.1:4000/metrics/，请求使用 `.env` 中的 `LITELLM_MASTER_KEY` 作为 Bearer 凭据

查看状态可运行 `docker compose --project-name litellm-local --env-file .env -f local-dev/compose.yaml ps`

停止服务可运行 `docker compose --project-name litellm-local --env-file .env -f local-dev/compose.yaml down`，数据库保存在 `litellm_local_postgres_data` 卷中

中文镜像基于官方稳定后端，修改克隆的 Python 源码后需要构建后端镜像才能生效。恢复官方英文界面时，直接使用 `local-dev/compose.yaml` 启动
