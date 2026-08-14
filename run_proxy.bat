@echo off
set PYTHONIOENCODING=utf-8
"C:\Users\pauls\AppData\Local\Programs\Python\Python313\Scripts\litellm.exe" --config "C:\Users\pauls\Documents\litellm\config.yaml" --port 4000 --detailed_debug > proxy_log.txt 2>&1
