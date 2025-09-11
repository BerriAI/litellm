# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from huggingface_hub import constants
from huggingface_hub._space_api import SpaceHardware
from huggingface_hub.utils._datetime import parse_datetime


class JobStage(str, Enum):
    """
    Enumeration of possible stage of a Job on the Hub.

    Value can be compared to a string:
    ```py
    assert JobStage.COMPLETED == "COMPLETED"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/job_types/JobInfo.ts#L61 (private url).
    """

    # Copied from moon-landing > server > lib > Job.ts
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    DELETED = "DELETED"
    RUNNING = "RUNNING"


@dataclass
class JobStatus:
    stage: JobStage
    message: Optional[str]


@dataclass
class JobOwner:
    id: str
    name: str
    type: str


@dataclass
class JobInfo:
    """
    Contains information about a Job.

    Args:
        id (`str`):
            Job ID.
        created_at (`datetime` or `None`):
            When the Job was created.
        docker_image (`str` or `None`):
            The Docker image from Docker Hub used for the Job.
            Can be None if space_id is present instead.
        space_id (`str` or `None`):
            The Docker image from Hugging Face Spaces used for the Job.
            Can be None if docker_image is present instead.
        command (`List[str]` or `None`):
            Command of the Job, e.g. `["python", "-c", "print('hello world')"]`
        arguments (`List[str]` or `None`):
            Arguments passed to the command
        environment (`Dict[str]` or `None`):
            Environment variables of the Job as a dictionary.
        secrets (`Dict[str]` or `None`):
            Secret environment variables of the Job (encrypted).
        flavor (`str` or `None`):
            Flavor for the hardware, as in Hugging Face Spaces. See [`SpaceHardware`] for possible values.
            E.g. `"cpu-basic"`.
        status: (`JobStatus` or `None`):
            Status of the Job, e.g. `JobStatus(stage="RUNNING", message=None)`
            See [`JobStage`] for possible stage values.
        status: (`JobOwner` or `None`):
            Owner of the Job, e.g. `JobOwner(id="5e9ecfc04957053f60648a3e", name="lhoestq", type="user")`

    Example:

    ```python
    >>> from huggingface_hub import run_job
    >>> job = run_job(
    ...     image="python:3.12",
    ...     command=["python", "-c", "print('Hello from the cloud!')"]
    ... )
    >>> job
    JobInfo(id='687fb701029421ae5549d998', created_at=datetime.datetime(2025, 7, 22, 16, 6, 25, 79000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='RUNNING', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq', type='user'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687fb701029421ae5549d998')
    >>> job.id
    '687fb701029421ae5549d998'
    >>> job.url
    'https://huggingface.co/jobs/lhoestq/687fb701029421ae5549d998'
    >>> job.status.stage
    'RUNNING'
    ```
    """

    id: str
    created_at: Optional[datetime]
    docker_image: Optional[str]
    space_id: Optional[str]
    command: Optional[List[str]]
    arguments: Optional[List[str]]
    environment: Optional[Dict[str, Any]]
    secrets: Optional[Dict[str, Any]]
    flavor: Optional[SpaceHardware]
    status: JobStatus
    owner: JobOwner

    # Inferred fields
    endpoint: str
    url: str

    def __init__(self, **kwargs) -> None:
        self.id = kwargs["id"]
        created_at = kwargs.get("createdAt") or kwargs.get("created_at")
        self.created_at = parse_datetime(created_at) if created_at else None
        self.docker_image = kwargs.get("dockerImage") or kwargs.get("docker_image")
        self.space_id = kwargs.get("spaceId") or kwargs.get("space_id")
        owner = kwargs.get("owner", {})
        self.owner = JobOwner(id=owner["id"], name=owner["name"], type=owner["type"])
        self.command = kwargs.get("command")
        self.arguments = kwargs.get("arguments")
        self.environment = kwargs.get("environment")
        self.secrets = kwargs.get("secrets")
        self.flavor = kwargs.get("flavor")
        status = kwargs.get("status", {})
        self.status = JobStatus(stage=status["stage"], message=status.get("message"))

        # Inferred fields
        self.endpoint = kwargs.get("endpoint", constants.ENDPOINT)
        self.url = f"{self.endpoint}/jobs/{self.owner.name}/{self.id}"
