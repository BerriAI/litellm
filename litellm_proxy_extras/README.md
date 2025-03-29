Additonal files for the proxy.

Separated from the main litellm package to reduce the size of the main package.

To use, run:

```bash
pip install litellm_proxy_extras
```

OR 

```bash
pip install litellm[proxy] # installs litellm_proxy_extras and other proxy dependencies.
```

Currently, the only extra is the migrations for the database.

To use the migrations, run:

```bash
litellm --use_prisma_migrate
```

