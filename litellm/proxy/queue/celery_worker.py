import os
from multiprocessing import Process


def run_worker(cwd):
    os.chdir(cwd)
    os.system(
        "celery -A celery_app.celery_app worker --concurrency=120 --loglevel=info"
    )


def start_worker(cwd):
    cwd += "/queue"
    worker_process = Process(target=run_worker, args=(cwd,))
    worker_process.start()
