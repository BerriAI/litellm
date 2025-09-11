"""@private"""

import os

common_release_envs = [
    # Render
    "RENDER_GIT_COMMIT",
    # GitLab CI
    "CI_COMMIT_SHA",
    # CircleCI
    "CIRCLE_SHA1",
    # Heroku
    "SOURCE_VERSION",
    # Travis CI
    "TRAVIS_COMMIT",
    # Jenkins (commonly used variable, but can be customized in Jenkins setups)
    "GIT_COMMIT",
    # GitHub Actions
    "GITHUB_SHA",
    # Bitbucket Pipelines
    "BITBUCKET_COMMIT",
    # Azure Pipelines
    "BUILD_SOURCEVERSION",
    # Drone CI
    "DRONE_COMMIT_SHA",
]


def get_common_release_envs():
    for env in common_release_envs:
        if env in os.environ:
            return os.environ[env]
    return None
