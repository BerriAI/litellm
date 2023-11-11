# import threading, time, litellm
# import concurrent.futures
# """
# v1: 

# 1. `--experimental_async` starts 2 background threads:
#     - 1. to check the redis queue:
#         - if job available 
#         - it dequeues as many jobs as healthy endpoints 
#         - calls llm api -> saves response in redis cache
#     - 2. to check the llm apis: 
#         - check if endpoints are healthy (unhealthy = 4xx / 5xx call or >1min. queue)
#         - which one is least busy 
# 2. /router/chat/completions: receives request -> adds to redis queue -> returns {run_id, started_at, request_obj}
# 3. /router/chat/completions/runs/{run_id}: returns {status: _, [optional] response_obj: _}
# """

# def _start_health_check_thread():
#     """
#     Starts a separate thread to perform health checks periodically.
#     """
#     health_check_thread = threading.Thread(target=_perform_health_checks, daemon=True)
#     health_check_thread.start()
#     llm_call_thread = threading.Thread(target=_llm_call_thread, daemon=True)
#     llm_call_thread.start()


# def _llm_call_thread():
#     """
#     Periodically performs job checks on the redis queue.
#     If available, make llm api calls. 
#     Write result to redis cache (1 min ttl)
#     """
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         while True: 
#             job_checks = _job_check() 
#             future_to_job = {executor.submit(_llm_api_call, job): job for job in job_checks}
#             for future in concurrent.futures.as_completed(future_to_job):
#                 job = future_to_job[future]
#                 try:
#                     result = future.result()
#                 except Exception as exc:
#                     print(f'{job} generated an exception: {exc}')
#                 else:
#                     _write_to_cache(job, result, ttl=1*60)
#             time.sleep(1)  # sleep 1 second to avoid overloading the server

        

# def _perform_health_checks():
#     """
#     Periodically performs health checks on the servers.
#     Updates the list of healthy servers accordingly.
#     """
#     while True:
#         healthy_deployments = _health_check()
#         # Adjust the time interval based on your needs
#         time.sleep(15)

# def _job_check(): 
#     """
#     Periodically performs job checks on the redis queue.
#     Returns the list of available jobs - len(available_jobs) == len(healthy_endpoints),
#     e.g. don't dequeue a gpt-3.5-turbo job if there's no healthy deployments left 
#     """
#     pass

# def _llm_api_call(**data):
#     """
#     Makes the litellm.completion() call with 3 retries 
#     """ 
#     return litellm.completion(num_retries=3, **data)

# def _write_to_cache(): 
#     """
#     Writes the result to a redis cache in the form (key:job_id, value: <response_object>) 
#     """ 
#     pass

# def _health_check():
#     """
#     Performs a health check on the deployments
#     Returns the list of healthy deployments
#     """
#     healthy_deployments = []
#     for deployment in model_list: 
#         litellm_args = deployment["litellm_params"]
#         try: 
#             start_time = time.time()
#             litellm.completion(messages=[{"role": "user", "content": ""}], max_tokens=1, **litellm_args) # hit the server with a blank message to see how long it takes to respond
#             end_time = time.time() 
#             response_time = end_time - start_time
#             logging.debug(f"response_time: {response_time}")
#             healthy_deployments.append((deployment, response_time))
#             healthy_deployments.sort(key=lambda x: x[1])
#         except Exception as e: 
#             pass
#     return healthy_deployments
