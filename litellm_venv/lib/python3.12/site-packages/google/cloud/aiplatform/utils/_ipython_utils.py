# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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
#

import sys
import typing
from uuid import uuid4
from typing import Optional

from google.cloud.aiplatform import base

if typing.TYPE_CHECKING:
    from google.cloud.aiplatform.metadata import experiment_resources
    from google.cloud.aiplatform import model_evaluation

_LOGGER = base.Logger(__name__)


def _get_ipython_shell_name() -> str:
    if "IPython" in sys.modules:
        from IPython import get_ipython

        return get_ipython().__class__.__name__
    return ""


def is_ipython_available() -> bool:
    return _get_ipython_shell_name() != ""


def _get_styles() -> None:
    """Returns the HTML style markup to support custom buttons."""
    return """
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    <style>
      .view-vertex-resource,
      .view-vertex-resource:hover,
      .view-vertex-resource:visited {
        position: relative;
        display: inline-flex;
        flex-direction: row;
        height: 32px;
        padding: 0 12px;
          margin: 4px 18px;
        gap: 4px;
        border-radius: 4px;

        align-items: center;
        justify-content: center;
        background-color: rgb(255, 255, 255);
        color: rgb(51, 103, 214);

        font-family: Roboto,"Helvetica Neue",sans-serif;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        text-decoration: none !important;

        transition: box-shadow 280ms cubic-bezier(0.4, 0, 0.2, 1) 0s;
        box-shadow: 0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12);
      }
      .view-vertex-resource:active {
        box-shadow: 0px 5px 5px -3px rgba(0,0,0,0.2),0px 8px 10px 1px rgba(0,0,0,0.14),0px 3px 14px 2px rgba(0,0,0,0.12);
      }
      .view-vertex-resource:active .view-vertex-ripple::before {
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        right: 0;
        border-radius: 4px;
        pointer-events: none;

        content: '';
        background-color: rgb(51, 103, 214);
        opacity: 0.12;
      }
      .view-vertex-icon {
        font-size: 18px;
      }
    </style>
  """


def display_link(text: str, url: str, icon: Optional[str] = "open_in_new") -> None:
    """Creates and displays the link to open the Vertex resource

    Args:
        text: The text displayed on the clickable button.
        url: The url that the button will lead to.
          Only cloud console URIs are allowed.
        icon: The icon name on the button (from material-icons library)

    Returns:
        Dict of custom properties with keys mapped to column names
    """
    CLOUD_UI_URL = "https://console.cloud.google.com"
    if not url.startswith(CLOUD_UI_URL):
        raise ValueError(f"Only urls starting with {CLOUD_UI_URL} are allowed.")

    button_id = f"view-vertex-resource-{str(uuid4())}"

    # Add the markup for the CSS and link component
    html = f"""
        {_get_styles()}
        <a class="view-vertex-resource" id="{button_id}" href="#view-{button_id}">
          <span class="material-icons view-vertex-icon">{icon}</span>
          <span>{text}</span>
        </a>
        """

    # Add the click handler for the link
    html += f"""
        <script>
          (function () {{
            const link = document.getElementById('{button_id}');
            link.addEventListener('click', (e) => {{
              if (window.google?.colab?.openUrl) {{
                window.google.colab.openUrl('{url}');
              }} else {{
                window.open('{url}', '_blank');
              }}
              e.stopPropagation();
              e.preventDefault();
            }});
          }})();
        </script>
    """

    from IPython.core.display import display
    from IPython.display import HTML

    display(HTML(html))


def display_experiment_button(experiment: "experiment_resources.Experiment") -> None:
    """Function to generate a link bound to the Vertex experiment"""
    if not is_ipython_available():
        return
    try:
        project = experiment._metadata_context.project
        location = experiment._metadata_context.location
        experiment_name = experiment._metadata_context.name
        if experiment_name is None or project is None or location is None:
            return
    except AttributeError:
        _LOGGER.warning("Unable to fetch experiment metadata")
        return

    uri = (
        "https://console.cloud.google.com/vertex-ai/experiments/locations/"
        + f"{location}/experiments/{experiment_name}/"
        + f"runs?project={project}"
    )
    display_link("View Experiment", uri, "science")


def display_model_evaluation_button(
    evaluation: "model_evaluation.ModelEvaluation",
) -> None:
    """Function to generate a link bound to the Vertex model evaluation"""
    if not is_ipython_available():
        return

    try:
        resource_name = evaluation.resource_name
        fields = evaluation._parse_resource_name(resource_name)
        project = fields["project"]
        location = fields["location"]
        model_id = fields["model"]
        evaluation_id = fields["evaluation"]
    except AttributeError:
        _LOGGER.warning("Unable to parse model evaluation metadata")
        return

    if "@" in model_id:
        model_id, version_id = model_id.split("@")
    else:
        version_id = "default"

    uri = (
        "https://console.cloud.google.com/vertex-ai/models/locations/"
        + f"{location}/models/{model_id}/versions/{version_id}/evaluations/"
        + f"{evaluation_id}?project={project}"
    )
    display_link("View Model Evaluation", uri, "model_training")
