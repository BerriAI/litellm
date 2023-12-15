from typing import List

from redis.client import Pipeline
from redis.exceptions import WatchError

from .job import Job


class Dependency:
    @classmethod
    def get_jobs_with_met_dependencies(cls, jobs: List['Job'], pipeline: Pipeline):
        jobs_with_met_dependencies = []
        jobs_with_unmet_dependencies = []
        for job in jobs:
            while True:
                try:
                    pipeline.watch(*[Job.key_for(dependency_id) for dependency_id in job._dependency_ids])
                    job.register_dependency(pipeline=pipeline)
                    if job.dependencies_are_met(pipeline=pipeline):
                        jobs_with_met_dependencies.append(job)
                    else:
                        jobs_with_unmet_dependencies.append(job)
                    pipeline.execute()
                except WatchError:
                    continue
                break
        return jobs_with_met_dependencies, jobs_with_unmet_dependencies
