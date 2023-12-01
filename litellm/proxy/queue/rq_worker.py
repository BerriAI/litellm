import sys, os 
from dotenv import load_dotenv
load_dotenv() 
# Add the path to the local folder to sys.path
sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path - for litellm local dev

def start_rq_worker(): 
    from rq import Worker, Queue, Connection
    from litellm._redis import get_redis_client
    # Set up RQ connection
    redis_conn = get_redis_client()
    print(redis_conn.ping())  # Should print True if connected successfully
    # Create a worker and add the queue
    try:
        queue = Queue(connection=redis_conn)
        worker = Worker([queue], connection=redis_conn)
    except Exception as e:
        print(f"Error setting up worker: {e}")
        exit()
    
    with Connection(redis_conn):
        worker.work()

start_rq_worker()