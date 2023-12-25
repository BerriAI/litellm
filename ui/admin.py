"""
Admin sets proxy url + allowed email subdomain 
"""
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64, os

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
    proxy_setup()
