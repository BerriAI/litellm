"""
Admin sets proxy url + allowed email subdomain 
"""
import streamlit as st

# Create a configuration placeholder
st.session_state.setdefault('proxy_url', 'http://example.com')
st.session_state.setdefault('allowed_email_subdomain', 'example.com')

# Simple function to update config values
def update_config():
    st.session_state['proxy_url'] = proxy_url
    st.session_state['allowed_email_subdomain'] = allowed_email_subdomain

# Display the form for the admin to set the proxy URL and allowed email subdomain
st.header("Admin Configuration")

with st.form("config_form", clear_on_submit=False):
    proxy_url = st.text_input("Set Proxy URL", st.session_state['proxy_url'])
    allowed_email_subdomain = st.text_input("Set Allowed Email Subdomain", st.session_state['allowed_email_subdomain'])
    submitted = st.form_submit_button("Save")

    if submitted:
        update_config()

# Display the current configuration
st.write(f"Current Proxy URL: {st.session_state['proxy_url']}")
st.write(f"Current Allowed Email Subdomain: {st.session_state['allowed_email_subdomain']}")