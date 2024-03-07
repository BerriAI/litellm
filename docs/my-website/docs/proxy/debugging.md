# Debugging

2 levels of debugging supported. 

- debug (prints info logs)
- detailed debug (prints debug logs)

## `debug`

**via cli**

```bash
$ litellm --debug
```

**via env**

```python
os.environ["LITELLM_LOG"] = "INFO"
```

## `detailed debug`

**via cli**

```bash
$ litellm --detailed_debug
```

**via env**

```python
os.environ["LITELLM_LOG"] = "DEBUG"
```