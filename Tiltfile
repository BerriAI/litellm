# Tiltfile
#
# for LiteLLM Proxy
#
# [Tilt](https://tilt.dev/) is a local development platform that makes it
# easy to develop applications for Kubernetes. Tilt automates all the
# steps from a code change to a new container image to a fresh pod.
#
# Tiltfiles are written in Starlark, a Python-inspired language.

load('ext://uibutton', 'cmd_button', 'text_input', 'choice_input', 'location')

secret_settings(disable_scrub=True)

docker_build(
    ref='litellm:some-tag',
    dockerfile='docker/Dockerfile.non_root',
    context='.',
)

# The helm built-in function lets you load from a chart on your filesystem.
#
# Calling helm() runs helm template on a chart directory and returns a blob of
# the Kubernetes YAML, which you can then deploy with k8s_yaml.
#
yaml = helm(
    './deploy/charts/litellm-helm',
    name='litellm-proxy',
    set=[
        "image.repository=litellm",
        "image.tag=some-tag",
        "masterkey=sk-master",
    ],
)
k8s_yaml(yaml)

k8s_resource(
    'litellm-proxy',
    # map one or more local ports to ports on your Pod; first number is local port, second is container port
    port_forwards=['4000:4000'],
)
