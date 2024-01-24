def add_new_model():
    import streamlit as st
    import json, requests, uuid

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
            f"Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page. Currently set Proxy Endpoint: {st.session_state.get('api_url', None)} and Proxy Key: {st.session_state.get('proxy_key', None)}"
        )


def create_key():
    import streamlit as st
    import json, requests, uuid

    if (
        st.session_state.get("api_url", None) is not None
        and st.session_state.get("proxy_key", None) is not None
    ):
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
                response = requests.post(
                    complete_url, json=key_post_body, headers=headers
                )

                if response.status_code == 200:
                    st.success(f"Key added successfully! - {response.json()}")
                else:
                    st.error(f"Failed to add Key. Status code: {response.status_code}")

                st.success("Form submitted successfully!")
            except Exception as e:
                raise e
    else:
        st.warning(
            f"Please configure the Proxy Endpoint and Proxy Key on the Proxy Setup page. Currently set Proxy Endpoint: {st.session_state.get('api_url', None)} and Proxy Key: {st.session_state.get('proxy_key', None)}"
        )


def streamlit_ui():
    import streamlit as st

    st.header("Admin Configuration")

    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ("Proxy Setup", "Add Models", "List Models", "Create Key")
    )

    # Initialize session state variables if not already present
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = None
    if "proxy_key" not in st.session_state:
        st.session_state["proxy_key"] = None

    # Display different pages based on navigation selection
    if page == "Proxy Setup":
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
    elif page == "Add Models":
        add_new_model()
    elif page == "List Models":
        list_models()
    elif page == "Create Key":
        create_key()


if __name__ == "__main__":
    streamlit_ui()
