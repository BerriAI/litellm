## Issue

The /v1/files/{file_id}/contents endpoint loads the file in memory before sending it to the client. During Load tests, multiple buffered files in memory lead to high peak memory usage and OOM

**Summary:**
During batch file retrievals with large payloads, worker memory rises sharply and quickly approaches the container limit before worker recycling can occur, indicating that large file responses are being buffered in memory rather than streamed.

**OOM Error is not reached** but container memory usage reaches peak 98 - 99%. Payload sizes of 65MB used with 1000 requests in parallel, according to the batch_lt.py script provided

Code Path (file_endpoints.py, line 716):
```
                response = await litellm.afile_content(
                    **{
                        "custom_llm_provider": custom_llm_provider,
                        "file_id": file_id,
                        **data,
                    }  # type: ignore
                )
``` 

**Steps**
- Ran **LiteLLM with 2 worker** and **with** `max_requests_before_restart`.
- Use **memray** (e.g. leak mode / `--leaks`) on the **files** path.
- **Mocked** file upstream / route so we can send a **~65 MB** payload through the proxy without relying on real provider keys, then attribute memory.


**Potential Resolution**
Without changing the current business logic, the potential fix could be:

 - Limitation of the openai SDK which returns BinaryResponse, instead of a streamed response.
 - We can wrap this API with a StreamingResponse handler, so that this can be mitigated. Again all signs point to this code path in the flamegraph as well.
 - Needs to be profiled after fix and before release.

### Before the fix

**Average Memory Usage**: 3.707 / 4 GiB (92.67%)
**Peak Memory Usage**: 3.893 / 4 GiB (97.32%)

Run Logs
```
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   34.76%    3.335GiB / 4GiB     83.37%    201GB / 201GB   269MB / 400MB   80
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   37.94%    3.58GiB / 4GiB      89.50%    206GB / 206GB   269MB / 400MB   83
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   8.29%     3.512GiB / 4GiB     87.81%    206GB / 206GB   269MB / 400MB   83
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   41.50%    3.656GiB / 4GiB     91.39%    214GB / 215GB   269MB / 400MB   83
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   14.97%    3.781GiB / 4GiB     94.52%    214GB / 215GB   269MB / 400MB   83
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   3.68%     3.495GiB / 4GiB     87.37%    236GB / 236GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   62.47%    3.69GiB / 4GiB      92.24%    236GB / 237GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   29.04%    3.687GiB / 4GiB     92.18%    240GB / 241GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   35.19%    3.883GiB / 4GiB     97.08%    243GB / 244GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   11.32%    3.819GiB / 4GiB     95.47%    243GB / 244GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   13.22%    3.685GiB / 4GiB     92.12%    244GB / 244GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   3.89%     3.822GiB / 4GiB     95.54%    244GB / 245GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   46.25%    3.822GiB / 4GiB     95.54%    244GB / 245GB   318MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   15.11%    3.768GiB / 4GiB     94.19%    246GB / 247GB   319MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   17.94%    3.822GiB / 4GiB     95.54%    246GB / 247GB   319MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   17.71%    3.766GiB / 4GiB     94.15%    251GB / 252GB   319MB / 400MB   82
khgokul@instance-20260329-212319:~/litellm-ifood-debug/litellm$ docker stats litellm-litellm-1 --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O       PIDS
c7505b5263d1   litellm-litellm-1   29.78%    3.893GiB / 4GiB     97.32%    252GB / 252GB   319MB / 400MB   82
```

### After the fix

Test: 1000 concurrent requests with a payload size of 65MB
**Average Memory Usage**: 2.56 / 4 GB (64%)
**Peak Memory Usage**: 2.621 / 4 GB (65.2%)


Run Logs
```
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O          BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   19.08%    2.455GiB / 4GiB     61.38%    9.89GB / 9.9GB   43.1MB / 21.1MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   20.34%    2.487GiB / 4GiB     62.18%    11.2GB / 11.3GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.11%    2.487GiB / 4GiB     62.18%    11.5GB / 11.6GB   43.1MB / 21.1MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.33%    2.492GiB / 4GiB     62.30%    11.7GB / 11.7GB   43.1MB / 21.1MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   32.51%    2.508GiB / 4GiB     62.70%    11.9GB / 11.9GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.88%    2.488GiB / 4GiB     62.19%    12GB / 12.1GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   27.51%    2.503GiB / 4GiB     62.56%    12.2GB / 12.3GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   33.72%    2.488GiB / 4GiB     62.19%    12.4GB / 12.4GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   32.68%    2.507GiB / 4GiB     62.68%    12.6GB / 12.6GB   43.1MB / 21.1MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   27.12%    2.513GiB / 4GiB     62.82%    12.8GB / 12.8GB   43.1MB / 21.1MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   26.59%    2.515GiB / 4GiB     62.87%    12.9GB / 12.9GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   31.63%    2.506GiB / 4GiB     62.64%    13.1GB / 13.1GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   19.79%    2.494GiB / 4GiB     62.35%    13.2GB / 13.3GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   56.66%    2.505GiB / 4GiB     62.64%    13.3GB / 13.4GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.55%    2.503GiB / 4GiB     62.57%    13.5GB / 13.5GB   43.1MB / 21.1MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.58%    2.499GiB / 4GiB     62.48%    13.7GB / 13.7GB   43.1MB / 21.1MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   44.50%    2.526GiB / 4GiB     63.15%    14.4GB / 14.4GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.61%    2.525GiB / 4GiB     63.11%    14.5GB / 14.5GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.20%    2.514GiB / 4GiB     62.85%    14.6GB / 14.7GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.55%    2.524GiB / 4GiB     63.09%    14.7GB / 14.8GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.63%    2.53GiB / 4GiB      63.26%    14.9GB / 14.9GB   43.1MB / 21.1MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   21.76%    2.556GiB / 4GiB     63.90%    18.7GB / 18.8GB   43.1MB / 21.1MB   53
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.33%    2.552GiB / 4GiB     63.81%    18.8GB / 18.9GB   43.1MB / 21.1MB   53
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.40%    2.556GiB / 4GiB     63.91%    19GB / 19.1GB   43.1MB / 21.1MB   53
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   29.29%    2.561GiB / 4GiB     64.03%    19.2GB / 19.3GB   43.1MB / 21.1MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   40.03%    2.56GiB / 4GiB      64.00%    19.5GB / 19.6GB   43.1MB / 21.1MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.49%    2.577GiB / 4GiB     64.41%    28.8GB / 29GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   28.65%    2.575GiB / 4GiB     64.37%    29.1GB / 29.3GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.35%    2.578GiB / 4GiB     64.46%    29.3GB / 29.5GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   19.23%    2.577GiB / 4GiB     64.44%    34GB / 34.4GB   43.1MB / 21.2MB   57
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.20%    2.577GiB / 4GiB     64.43%    34.2GB / 34.5GB   43.1MB / 21.2MB   57
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   27.50%    2.595GiB / 4GiB     64.88%    37.1GB / 37.5GB   43.1MB / 21.2MB   57
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.41%    2.588GiB / 4GiB     64.71%    37.3GB / 37.7GB   43.1MB / 21.2MB   57
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   38.88%    2.607GiB / 4GiB     65.17%    38.7GB / 39.1GB   43.1MB / 21.2MB   57
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   30.39%    2.606GiB / 4GiB     65.15%    38.8GB / 39.2GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   37.61%    2.597GiB / 4GiB     64.92%    39GB / 39.4GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   43.09%    2.558GiB / 4GiB     63.94%    39.1GB / 39.5GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   43.19%    2.6GiB / 4GiB       65.01%    39.3GB / 39.7GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   26.15%    2.584GiB / 4GiB     64.61%    39.4GB / 39.8GB   43.1MB / 21.2MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   29.18%    2.594GiB / 4GiB     64.84%    42.2GB / 42.6GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   30.01%    2.595GiB / 4GiB     64.88%    42.3GB / 42.8GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   30.10%    2.604GiB / 4GiB     65.10%    43.4GB / 43.8GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   37.32%    2.597GiB / 4GiB     64.93%    43.6GB / 44.1GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   78.75%    2.613GiB / 4GiB     65.32%    44.9GB / 45.4GB   43.1MB / 21.2MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   32.54%    2.615GiB / 4GiB     65.37%    45.1GB / 45.5GB   43.1MB / 21.2MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.97%    2.611GiB / 4GiB     65.26%    45.2GB / 45.7GB   43.1MB / 21.2MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.59%    2.613GiB / 4GiB     65.33%    45.5GB / 46GB   43.1MB / 21.2MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   21.86%    2.596GiB / 4GiB     64.90%    48.5GB / 49GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   25.42%    2.59GiB / 4GiB      64.75%    48.7GB / 49.2GB   43.1MB / 21.2MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   30.47%    2.615GiB / 4GiB     65.38%    52.6GB / 53.1GB   43.1MB / 21.3MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   24.78%    2.613GiB / 4GiB     65.32%    52.7GB / 53.3GB   43.1MB / 21.3MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   20.47%    2.617GiB / 4GiB     65.42%    52.8GB / 53.4GB   43.1MB / 21.3MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   22.36%    2.603GiB / 4GiB     65.09%    53GB / 53.5GB   43.1MB / 21.3MB   53
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   27.24%    2.606GiB / 4GiB     65.16%    53.1GB / 53.7GB   43.1MB / 21.3MB   54
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   20.96%    2.624GiB / 4GiB     65.60%    57.7GB / 58.3GB   43.1MB / 21.3MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   18.32%    2.603GiB / 4GiB     65.07%    57.8GB / 58.5GB   43.1MB / 21.3MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   18.01%    2.621GiB / 4GiB     65.53%    61.9GB / 62.5GB   43.1MB / 21.3MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O         BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   14.86%    2.602GiB / 4GiB     65.04%    62GB / 62.7GB   43.1MB / 21.3MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   21.48%    2.611GiB / 4GiB     65.29%    62.4GB / 63.1GB   43.1MB / 21.3MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   19.22%    2.619GiB / 4GiB     65.48%    62.5GB / 63.2GB   43.1MB / 21.3MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   19.12%    2.62GiB / 4GiB      65.51%    67.5GB / 68.3GB   43.1MB / 21.3MB   55
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   21.15%    2.618GiB / 4GiB     65.44%    67.6GB / 68.4GB   43.1MB / 21.3MB   56
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   0.87%     2.496GiB / 4GiB     62.39%    68.5GB / 69.4GB   43.1MB / 21.3MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   12.21%    2.496GiB / 4GiB     62.39%    68.5GB / 69.4GB   43.1MB / 21.3MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   0.79%     2.491GiB / 4GiB     62.28%    68.5GB / 69.4GB   44.4MB / 21.4MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   0.80%     2.491GiB / 4GiB     62.28%    68.5GB / 69.4GB   44.4MB / 21.4MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   0.79%     2.491GiB / 4GiB     62.28%    68.5GB / 69.4GB   44.4MB / 21.4MB   52
(.venv) harishgokul@instance-20260404-194023:~/litellm$ docker compose stats litellm --no-stream
CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
93a259999492   litellm-litellm-1   0.55%     2.491GiB / 4GiB     62.28%    68.5GB / 69.4GB   44.4MB / 21.4MB   52
```