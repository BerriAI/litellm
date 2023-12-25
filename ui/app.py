"""
Routes between admin, auth, keys pages
"""
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64, binascii, os
from admin import admin_page
from auth import auth_page
from urllib.parse import urlparse, parse_qs


# Parse the query params in the URL
def get_query_params():
    # Get the query params from Streamlit's `server.request` function
    # This functionality is not officially documented and could change in the future versions of Streamlit
    query_params = st.experimental_get_query_params()
    return query_params


def is_base64(sb):
    try:
        if isinstance(sb, str):
            # Try to encode it to bytes if it's a unicode string
            sb_bytes = sb.encode("ascii")
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            # If it is not a byte or a string, it is not base64
            return False

        # Check if decoding is successful.
        # The result of the decode is not required, so it is ignored.
        _ = base64.urlsafe_b64decode(sb_bytes)

        # If the decode was successful, the input is likely base64
        return True
    except (binascii.Error, ValueError):
        # If an error occurs, return False, as the input is not base64
        return False


# Check the URL path and route to the correct page based on the path
query_params = get_query_params()
page_param = query_params.get("page", [None])[0]

# Route to the appropriate page based on the URL query param
if page_param:
    try:
        # Try to decode the page_param from base64
        if is_base64(page_param):
            auth_page(redirect_url=f"{os.getenv('BASE_URL')}/{page_param}")
        else:
            st.error("Unknown page")
    except Exception as e:
        st.error("Failed to decode the page parameter. Error: " + str(e))
else:
    admin_page()
