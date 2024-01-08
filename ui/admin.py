"""
Admin sets proxy url + allowed email subdomain 
"""
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64, os, json, uuid, requests

# Replace your_base_url with the actual URL where the proxy auth app is hosted
your_base_url = os.getenv("BASE_URL")  # Example base URL


# Function to encode the configuration
def encode_config(proxy_url, allowed_email_subdomain, admin_emails):
    combined_string = f"proxy_url={proxy_url}&accepted_email_subdomain={allowed_email_subdomain}&admin_emails={admin_emails}"
    return base64.b64encode(combined_string.encode("utf-8")).decode("utf-8")


# Simple function to update config values
def update_config(proxy_url, allowed_email_subdomain, admin_emails):
    st.session_state["proxy_url"] = proxy_url
    st.session_state["allowed_email_subdomain"] = allowed_email_subdomain
    st.session_state["admin_emails"] = admin_emails
    if your_base_url.endswith("/"):
        st.session_state[
            "user_auth_url"
        ] = f"{your_base_url}user?page={encode_config(proxy_url=proxy_url, allowed_email_subdomain=allowed_email_subdomain, admin_emails=admin_emails)}"
    else:
        st.session_state[
            "user_auth_url"
        ] = f"{your_base_url}/user?page={encode_config(proxy_url=proxy_url, allowed_email_subdomain=allowed_email_subdomain, admin_emails=admin_emails)}"
    st.session_state["is_admin"] = True


def proxy_setup():
    # Create a configuration placeholder
    st.session_state.setdefault("proxy_url", "http://example.com")
    st.session_state.setdefault("allowed_email_subdomain", "example.com")
    st.session_state.setdefault("admin_emails", "admin@example.com")
    st.session_state.setdefault("user_auth_url", "NOT_GIVEN")

    with st.form("config_form", clear_on_submit=False):
        proxy_url = st.text_input("Set Proxy URL", st.session_state["proxy_url"])
        allowed_email_subdomain = st.text_input(
            "Set Allowed Email Subdomain", st.session_state["allowed_email_subdomain"]
        )
        admin_emails = st.text_input(
            "Allowed Admin Emails (add ',' to separate multiple emails)",
            st.session_state["admin_emails"],
        )
        submitted = st.form_submit_button("Save")

        if submitted:
            update_config(
                proxy_url=proxy_url,
                allowed_email_subdomain=allowed_email_subdomain,
                admin_emails=admin_emails,
            )

    # Display the current configuration
    st.write(f"Current Proxy URL: {st.session_state['proxy_url']}")
    st.write(
        f"Current Allowed Email Subdomain: {st.session_state['allowed_email_subdomain']}"
    )
    st.write(f"Current User Auth URL: {st.session_state['user_auth_url']}")


def add_new_model():
    import streamlit as st
    import json, requests, uuid

    if (
        st.session_state.get("api_url", None) is None
        and st.session_state.get("proxy_key", None) is None
    ):
        st.warning(
            "Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page."
        )

    model_name = st.text_input(
        "Model Name - user-facing model name", placeholder="gpt-3.5-turbo"
    )
    st.subheader("LiteLLM Params")
    litellm_model_name = st.text_input(
        "Model", placeholder="azure/gpt-35-turbo-us-east"
    )
    litellm_api_key = st.text_input("API Key")
    litellm_api_base = st.text_input(
        "API Base",
        placeholder="https://my-endpoint.openai.azure.com",
    )
    litellm_api_version = st.text_input("API Version", placeholder="2023-07-01-preview")
    litellm_params = json.loads(
        st.text_area(
            "Additional Litellm Params (JSON dictionary). [See all possible inputs](https://github.com/BerriAI/litellm/blob/3f15d7230fe8e7492c95a752963e7fbdcaf7bf98/litellm/main.py#L293)",
            value={},
        )
    )
    st.subheader("Model Info")
    mode_options = ("completion", "embedding", "image generation")
    mode_selected = st.selectbox("Mode", mode_options)
    model_info = json.loads(
        st.text_area(
            "Additional Model Info (JSON dictionary)",
            value={},
        )
    )

    if st.button("Submit"):
        try:
            model_info = {
                "model_name": model_name,
                "litellm_params": {
                    "model": litellm_model_name,
                    "api_key": litellm_api_key,
                    "api_base": litellm_api_base,
                    "api_version": litellm_api_version,
                },
                "model_info": {
                    "id": str(uuid.uuid4()),
                    "mode": mode_selected,
                },
            }
            # Make the POST request to the specified URL
            complete_url = ""
            if st.session_state["api_url"].endswith("/"):
                complete_url = f"{st.session_state['api_url']}model/new"
            else:
                complete_url = f"{st.session_state['api_url']}/model/new"

            headers = {"Authorization": f"Bearer {st.session_state['proxy_key']}"}
            response = requests.post(complete_url, json=model_info, headers=headers)

            if response.status_code == 200:
                st.success("Model added successfully!")
            else:
                st.error(f"Failed to add model. Status code: {response.status_code}")

            st.success("Form submitted successfully!")
        except Exception as e:
            raise e


def list_models():
    import streamlit as st
    import requests

    # Check if the necessary configuration is available
    if (
        st.session_state.get("api_url", None) is not None
        and st.session_state.get("proxy_key", None) is not None
    ):
        # Make the GET request
        try:
            complete_url = ""
            if isinstance(st.session_state["api_url"], str) and st.session_state[
                "api_url"
            ].endswith("/"):
                complete_url = f"{st.session_state['api_url']}models"
            else:
                complete_url = f"{st.session_state['api_url']}/models"
            response = requests.get(
                complete_url,
                headers={"Authorization": f"Bearer {st.session_state['proxy_key']}"},
            )
            # Check if the request was successful
            if response.status_code == 200:
                models = response.json()
                st.write(models)  # or st.json(models) to pretty print the JSON
            else:
                st.error(f"Failed to get models. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"An error occurred while requesting models: {e}")
    else:
        st.warning(
            "Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page."
        )


def create_key():
    import streamlit as st
    import json, requests, uuid

    if (
        st.session_state.get("api_url", None) is None
        and st.session_state.get("proxy_key", None) is None
    ):
        st.warning(
            "Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page."
        )

    duration = st.text_input("Duration - Can be in (h,m,s)", placeholder="1h")

    models = st.text_input("Models it can access (separated by comma)", value="")
    models = models.split(",") if models else []

    additional_params = json.loads(
        st.text_area(
            "Additional Key Params (JSON dictionary). [See all possible inputs](https://litellm-api.up.railway.app/#/key%20management/generate_key_fn_key_generate_post)",
            value={},
        )
    )

    if st.button("Submit"):
        try:
            key_post_body = {
                "duration": duration,
                "models": models,
                **additional_params,
            }
            # Make the POST request to the specified URL
            complete_url = ""
            if st.session_state["api_url"].endswith("/"):
                complete_url = f"{st.session_state['api_url']}key/generate"
            else:
                complete_url = f"{st.session_state['api_url']}/key/generate"

            headers = {"Authorization": f"Bearer {st.session_state['proxy_key']}"}
            response = requests.post(complete_url, json=key_post_body, headers=headers)

            if response.status_code == 200:
                st.success(f"Key added successfully! - {response.json()}")
            else:
                st.error(f"Failed to add Key. Status code: {response.status_code}")

            st.success("Form submitted successfully!")
        except Exception as e:
            raise e


def update_config():
    if (
        st.session_state.get("api_url", None) is None
        and st.session_state.get("proxy_key", None) is None
    ):
        st.warning(
            "Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page."
        )

    st.markdown("#### Alerting")
    input_slack_webhook = st.text_input(
        "Slack Webhook URL (Optional)",
        value=st.session_state.get("slack_webhook", ""),
        placeholder="https://hooks.slack.com/services/...",
    )
    st.markdown(
        "More information on Slack alerting configuration can be found in the [documentation]"
        "(https://docs.litellm.ai/docs/proxy/alerting)."
    )
    alerting_threshold = st.text_input(
        "Alerting threshold (in seconds) (Optional)",
        value=st.session_state.get("alerting_threshold", 300),
        placeholder=300,
    )
    st.markdown("How long to wait before a request is considered hanging")
    st.markdown("#### Logging")

    enable_langfuse_logging = st.checkbox("Enable Langfuse Logging")
    if enable_langfuse_logging == True:
        langfuse_host_url = st.text_input(
            "Langfuse Host",
            value=st.session_state.get("langfuse_host", "https://cloud.langfuse.com"),
            placeholder="https://cloud.langfuse.com",
        )
        langfuse_public_key = st.text_input(
            "Langfuse Public Key",
            value=st.session_state.get("langfuse_public_key", ""),
            placeholder="pk-lf-...",
        )
        langfuse_secret_key = st.text_input(
            "Langfuse Secret Key",
            value=st.session_state.get("langfuse_secret_key", ""),
            placeholder="sk-lf-...",
        )
    # When the "Save" button is clicked, update the session state
    if st.button("Save"):
        try:
            config_post_body = {}
            if (
                enable_langfuse_logging == True
                and langfuse_host_url is not None
                and langfuse_public_key is not None
                and langfuse_secret_key is not None
            ):
                config_post_body["litellm_settings"] = {
                    "success_callback": ["langfuse"]
                }
                config_post_body["environment_variables"] = {
                    "LANGFUSE_HOST": langfuse_host_url,
                    "LANGFUSE_PUBLIC_KEY": langfuse_public_key,
                    "LANGFUSE_SECRET_KEY": langfuse_secret_key,
                }
            if input_slack_webhook is not None and alerting_threshold is not None:
                config_post_body["general_settings"] = {
                    "alerting": ["slack"],
                    "alerting_threshold": alerting_threshold,
                }
                config_post_body["environment_variables"] = {
                    "SLACK_WEBHOOK_URL": input_slack_webhook
                }

            # Make the POST request to the specified URL
            complete_url = ""
            if st.session_state["api_url"].endswith("/"):
                complete_url = f"{st.session_state.get('api_url')}config/update"
            else:
                complete_url = f"{st.session_state.get('api_url')}/config/update"

            headers = {"Authorization": f"Bearer {st.session_state['proxy_key']}"}
            response = requests.post(
                complete_url, json=config_post_body, headers=headers
            )

            if response.status_code == 200:
                st.success(f"Config updated successfully! - {response.json()}")
            else:
                st.error(
                    f"Failed to update config. Status code: {response.status_code}. Error message: {response.json()['detail']}"
                )

            st.success("Form submitted successfully!")
        except Exception as e:
            raise e


def admin_page(is_admin="NOT_GIVEN"):
    # Display the form for the admin to set the proxy URL and allowed email subdomain
    st.header("Admin Configuration")
    st.session_state.setdefault("is_admin", is_admin)
    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Connect to Proxy",
            "Update Config",
            "Add Models",
            "List Models",
            "Create Key",
            "End-User Auth",
        ),
    )
    # Display different pages based on navigation selection
    if page == "Connect to Proxy":
        # Use text inputs with intermediary variables
        input_api_url = st.text_input(
            "Proxy Endpoint",
            value=st.session_state.get("api_url", ""),
            placeholder="http://0.0.0.0:8000",
        )
        input_proxy_key = st.text_input(
            "Proxy Key",
            value=st.session_state.get("proxy_key", ""),
            placeholder="sk-...",
        )
        # When the "Save" button is clicked, update the session state
        if st.button("Save"):
            st.session_state["api_url"] = input_api_url
            st.session_state["proxy_key"] = input_proxy_key
            st.success("Configuration saved!")
    elif page == "Update Config":
        update_config()
    elif page == "End-User Auth":
        proxy_setup()
    elif page == "Add Models":
        add_new_model()
    elif page == "List Models":
        list_models()
    elif page == "Create Key":
        create_key()


admin_page()
