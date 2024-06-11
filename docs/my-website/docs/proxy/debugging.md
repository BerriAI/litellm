# Debugging

2 levels of debugging supported. 

- debug (prints info logs)
- detailed debug (prints debug logs)

The proxy also supports json logs. [See here](#json-logs)

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

## JSON LOGS

Set `JSON_LOGS="True"` in your env:

```bash
export JSON_LOGS="True"
```
**OR**

Set `json_logs: true` in your yaml: 

```yaml
litellm_settings:
    json_logs: true
```

Start proxy 

```bash
$ litellm
```

The proxy will now all logs in json format.

## Control Log Output 

Turn off fastapi's default 'INFO' logs 

1. Turn on 'json logs' 
```yaml
litellm_settings:
    json_logs: true
```

2. Set `LITELLM_LOG` to 'ERROR' 

Only get logs if an error occurs. 

```bash
LITELLM_LOG="ERROR"
```

3. Start proxy 


```bash
$ litellm
```

Expected Output: 

```bash
# no info statements
```