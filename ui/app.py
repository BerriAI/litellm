"""
Routes between admin, auth, keys pages
"""
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64, binascii, os, json
from admin import admin_page
from auth import auth_page, verify_with_otp
import urllib.parse


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
        decoded_params = base64.urlsafe_b64decode(sb_bytes)

        # If the decode was successful, the input is likely base64
        return True, decoded_params
    except (binascii.Error, ValueError):
        # If an error occurs, return False, as the input is not base64
        return False


# Check the URL path and route to the correct page based on the path
query_params = get_query_params()
print(f"query_params: {query_params}")
page_param = query_params.get("page", [None])[0]
token_hash = query_params.get("token_hash", [None])[0]
decoded_token = None
if token_hash is not None:
    print(f"token_hash: {token_hash}")
    decoded_token = verify_with_otp(token=token_hash)
    print(f"decoded_token: {decoded_token}")
if page_param is not None:
    try:
        print(f"page_param: {page_param}")
        # Try to decode the page_param from base64
        is_valid, decoded_params = is_base64(page_param)
        print(f"is_valid: {is_valid}; decoded_params: {decoded_params}")
        if is_valid:
            if decoded_token is None:
                auth_page(page_param=page_param)
            else:
                # Convert the bytes to a string
                params_str = decoded_params.decode("utf-8")

                # Parse the parameters
                params = urllib.parse.parse_qs(params_str)

                # Extract the value of admin_emails
                admin_emails = params.get("admin_emails", [""])[0].split(",")

                print(admin_emails)
                print(vars(decoded_token.user))
                if decoded_token.user.email in admin_emails:
                    # admin ui
                    admin_page(is_admin=True)
                else:
                    # user ui
                    st.write(
                        f"email: {decoded_token.user.email}; admin_emails: {admin_emails}"
                    )
        else:
            st.error("Unknown page")
    except Exception as e:
        st.error("Failed to decode the page parameter. Error: " + str(e))
else:
    admin_page(is_admin=False)
