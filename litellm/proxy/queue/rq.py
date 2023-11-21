import os
import subprocess
import sys
import multiprocessing
from dotenv import load_dotenv
load_dotenv()

def run_rq_worker(redis_url):
    command = ["rq", "worker", "--url", redis_url]
    subprocess.run(command)

def start_rq_worker_in_background():
    # Set OBJC_DISABLE_INITIALIZE_FORK_SAFETY to YES
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    # Check if required environment variables are set
    required_vars = ["REDIS_USERNAME", "REDIS_PASSWORD", "REDIS_HOST", "REDIS_PORT"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        print(f"Error: Redis environment variables not set. Please set {', '.join(missing_vars)}.")
        sys.exit(1)

    # Construct Redis URL
    REDIS_URL = f"redis://{os.environ['REDIS_USERNAME']}:{os.environ['REDIS_PASSWORD']}@{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}"

    # Run rq worker in a separate process
    worker_process = multiprocessing.Process(target=run_rq_worker, args=(REDIS_URL,))
    worker_process.start()

if __name__ == "__main__":
    start_rq_worker_in_background()
