import os
from multiprocessing import Process

def run_worker():
    os.system("celery worker -A your_project_name.celery_app --concurrency=10 --loglevel=info")

if __name__ == "__main__":
    worker_process = Process(target=run_worker)
    worker_process.start()