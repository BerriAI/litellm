Additional files for the proxy. Reduces the size of the main litellm package.

Currently, only stores the migration.sql files for litellm-proxy.

To install, run:

```bash
pip install litellm-proxy-extras
```
OR 

```bash
pip install litellm[proxy] # installs litellm-proxy-extras and other proxy dependencies
```

To use the migrations, run:

```bash
litellm --use_prisma_migrate
```

