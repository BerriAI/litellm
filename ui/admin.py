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
    st.session_state[
        "user_auth_url"
    ] = f"{your_base_url}/?page={encode_config(proxy_url=proxy_url, allowed_email_subdomain=allowed_email_subdomain, admin_emails=admin_emails)}"
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


def admin_page(is_admin="NOT_GIVEN"):
    # Display the form for the admin to set the proxy URL and allowed email subdomain
    st.header("Admin Configuration")
    st.session_state.setdefault("is_admin", is_admin)
    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Proxy Setup", "Add Models"))
    # Display different pages based on navigation selection
    if page == "Proxy Setup":
        proxy_setup()
    elif page == "Add Models":
        if st.session_state["is_admin"] != True:
            st.write("Complete Proxy Setup to add new models")
        else:
            proxy_key = st.text_input("Proxy Key", placeholder="sk-...")
            model_name = st.text_input(
                "Model Name - user-facing model name", placeholder="gpt-3.5-turbo"
            )
            litellm_model_name = st.text_input(
                "LiteLLM Model Name", placeholder="azure/gpt-35-turbo-us-east"
            )
            litellm_api_key = st.text_input("LiteLLM API Key")
            litellm_api_base = st.text_input(
                "[Optional] LiteLLM API Base",
                placeholder="https://my-endpoint.openai.azure.com",
            )
            litellm_api_version = st.text_input(
                "[Optional] LiteLLM API Version", placeholder="2023-07-01-preview"
            )
            litellm_params = json.loads(
                st.text_area(
                    "Additional Litellm Params (JSON dictionary). [See all possible inputs](https://github.com/BerriAI/litellm/blob/3f15d7230fe8e7492c95a752963e7fbdcaf7bf98/litellm/main.py#L293)",
                    value={},
                )
            )
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
                    if st.session_state["proxy_url"].endswith("/"):
                        complete_url = f"{st.session_state['proxy_url']}model/new"
                    else:
                        complete_url = f"{st.session_state['proxy_url']}/model/new"

                    headers = {"Authorization": f"Bearer {proxy_key}"}
                    response = requests.post(
                        complete_url, json=model_info, headers=headers
                    )

                    if response.status_code == 200:
                        st.success("Model added successfully!")
                    else:
                        st.error(
                            f"Failed to add model. Status code: {response.status_code}"
                        )

                    st.success("Form submitted successfully!")
                except Exception as e:
                    raise e
