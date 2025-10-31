## 环境初始化

### 后端

```shell
pyenv shell 3.12.7
# venv创建在项目中
poetry config virtualenvs.in-project true
# poetry env list  # shows the name of the current environment
# poetry env remove <current environment>
# 添加依赖
# pipx install --suffix=@1.8.3 poetry==1.8.3
# poetry@1.8.3 lock
# make install-proxy-dev

POETRY_VIRTUALENVS_IN_PROJECT=true && poetry install --with dev --extras proxy
poetry env info

pip install detect_secrets
pip install google.auth

# eval $(poetry env activate)
# pip install -e ".[all]"
# python db_scripts/create_views.py
# export PROXY_BASE_URL="http://localhost:3000/"
python litellm/proxy/proxy_cli.py --config ./proxy_server_config_dev.yaml --port 4000
```

### 前端

```shell
export NODE_ENV="development"
cd ui/litellm-dashboard
npm run dev
# open http://localhost:3000/ui/
```
