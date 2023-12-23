"""
Admin sets proxy url + allowed email subdomain 
"""
import streamlit as st
import base64

# Replace your_base_url with the actual URL where the proxy auth app is hosted
your_base_url = 'http://localhost:8501'  # Example base URL

# Function to encode the configuration
def encode_config(proxy_url, allowed_email_subdomain):
    combined_string = f"proxy_url={proxy_url}&accepted_email_subdomain={allowed_email_subdomain}"
    return base64.b64encode(combined_string.encode('utf-8')).decode('utf-8')

# Simple function to update config values
def update_config(proxy_url, allowed_email_subdomain):
    st.session_state['proxy_url'] = proxy_url
    st.session_state['allowed_email_subdomain'] = allowed_email_subdomain
    st.session_state['user_auth_url'] = f"{your_base_url}/?page={encode_config(proxy_url=proxy_url, allowed_email_subdomain=allowed_email_subdomain)}"

def admin_page():
    # Display the form for the admin to set the proxy URL and allowed email subdomain
    st.header("Admin Configuration")
    # Create a configuration placeholder
    st.session_state.setdefault('proxy_url', 'http://example.com')
    st.session_state.setdefault('allowed_email_subdomain', 'example.com')
    st.session_state.setdefault('user_auth_url', 'NOT_GIVEN')

    with st.form("config_form", clear_on_submit=False):
        proxy_url = st.text_input("Set Proxy URL", st.session_state['proxy_url'])
        allowed_email_subdomain = st.text_input("Set Allowed Email Subdomain", st.session_state['allowed_email_subdomain'])
        submitted = st.form_submit_button("Save")

        if submitted:
            update_config(proxy_url=proxy_url, allowed_email_subdomain=allowed_email_subdomain)

    # Display the current configuration
    st.write(f"Current Proxy URL: {st.session_state['proxy_url']}")
    st.write(f"Current Allowed Email Subdomain: {st.session_state['allowed_email_subdomain']}")
    st.write(f"Current User Auth URL: {st.session_state['user_auth_url']}")