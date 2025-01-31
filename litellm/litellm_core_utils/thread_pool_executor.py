from concurrent.futures import ThreadPoolExecutor

MAX_THREADS = 100
# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
