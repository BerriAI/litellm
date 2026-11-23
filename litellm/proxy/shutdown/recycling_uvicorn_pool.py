import logging
from typing import Final

from uvicorn import Server
from uvicorn import config as uvicorn_config
from uvicorn.supervisors.multiprocess import Multiprocess, Process

from litellm.proxy.shutdown.recycling_uvicorn_server import RecyclingUvicornServer

LOGGER: Final = logging.getLogger("uvicorn.error")


class RecyclingProcess(Process):
    @property
    def server(self) -> Server:
        if self._server is None:
            self._server = RecyclingUvicornServer(self.config)
        return self._server


class RecyclingWorkerPool(Multiprocess):
    processes_num: int

    def _start_process(self) -> RecyclingProcess:
        process: Final = RecyclingProcess(self.config, self.sockets)
        process.start()
        return process

    def init_processes(self) -> None:
        self.processes.extend(self._start_process() for _ in range(self.processes_num))

    def handle_ttin(self) -> None:
        LOGGER.info("Received SIGTTIN, increasing the number of processes.")
        self.processes_num += 1
        self.processes.append(self._start_process())

    def restart_all(self) -> None:
        for index, old in enumerate(self.processes):
            if self.should_exit.is_set():
                return
            if not (replacement := self._start_process()).wait_until_ready(
                self.config.timeout_worker_healthcheck, self.should_exit
            ):
                replacement.kill()
                replacement.join()
                if not self.should_exit.is_set():
                    LOGGER.error(
                        "New child process [%s] was not ready in time, keeping worker [%s]", replacement.pid, old.pid
                    )
                return
            old.terminate()
            old.join()
            self.processes[index] = replacement

    def keep_subprocess_alive(self) -> None:
        if self.should_exit.is_set():
            return
        for index, process in enumerate(self.processes):
            if process.is_alive(timeout=self.config.timeout_worker_healthcheck):
                continue
            process.kill()
            process.join()
            if process.exitcode == uvicorn_config.STARTUP_FAILURE:
                LOGGER.error("Child process [%s] failed to start, stopping the parent process.", process.pid)
                self.should_exit.set()
                return
            if self.should_exit.is_set():
                return
            LOGGER.info("Child process [%s] died", process.pid)
            self.processes[index] = self._start_process()
