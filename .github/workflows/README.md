# Simple PyPI Publishing

A GitHub workflow to manually publish LiteLLM packages to PyPI with a specified version.

## How to Use

1. Go to the **Actions** tab in the GitHub repository
2. Select **Simple PyPI Publish** from the workflow list
3. Click **Run workflow**
4. Enter the version to publish (e.g., `1.74.10`)

## What the Workflow Does

1. **Updates** the version in `pyproject.toml`
2. **Copies** the model prices backup file
3. **Builds** the Python package
4. **Publishes** to PyPI

## Prerequisites

Make sure the following secret is configured in the repository:
- `PYPI_PUBLISH_PASSWORD`: PyPI API token for authentication

## Example Usage

- Version: `1.74.11` → Publishes as v1.74.11
- Version: `1.74.10-hotfix1` → Publishes as v1.74.10-hotfix1

## Features

- ✅ Manual trigger with version input
- ✅ Automatic version updates in `pyproject.toml`
- ✅ Repository safety check (only runs on official repo)
- ✅ Clean package building and publishing
- ✅ Success confirmation with PyPI package link 