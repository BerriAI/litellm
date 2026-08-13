from concurrent.futures import ThreadPoolExecutor
from typing import Final

MAX_THREADS: Final = 100
# Create a ThreadPoolExecutor
executor: Final = ThreadPoolExecutor(max_workers=MAX_THREADS)
