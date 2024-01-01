"""
Auth in user, to proxy ui. 

Uses supabase passwordless auth: https://supabase.com/docs/reference/python/auth-signinwithotp

Remember to set your redirect url to 8501 (streamlit default).
"""
import streamlit as st
from dotenv import load_dotenv
import requests, base64, binascii

load_dotenv()
import os


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
        params_str = decoded_params.decode("utf-8")
        param_dict = {}
        # split on the &
        params = params_str.split("&")
        # split on the =
        for param in params:
            split_val = param.split("=")
            param_dict[split_val[0].strip()] = split_val[1].strip()
        # If the decode was successful, the input is likely base64
        return True, param_dict
    except (binascii.Error, ValueError):
        # If an error occurs, return False, as the input is not base64
        return False


def sign_in_with_otp(email: str, page_param: str):
    print(f"received page param: {page_param}")
    b64_flag, decoded_params = is_base64(sb=page_param)
    print(f"b64_flag: {b64_flag}")
    st.write(f"Decoded params: {decoded_params}")
    # requests.post()
    # data = supabase.auth.sign_in_with_otp(
    #     {"email": email, "options": {"data": {"page_param": page_param}}}
    # )
    # print(f"data: {data}")
    # # Redirect to Supabase UI with the return data
    # st.write(f"Please check your email for a login link!")


# def verify_with_otp(token: str):
#     res = supabase.auth.verify_otp({"token_hash": token, "type": "email"})
#     return res


# Create the Streamlit app
def auth_page(page_param: str):
    st.title("User Authentication")

    # User email input
    email = st.text_input("Enter your email")

    # Sign in button
    if st.button("Sign In"):
        b64_flag, decoded_params = is_base64(sb=page_param)
        # Define the endpoint you want to make a POST request to
        if decoded_params["proxy_url"].endswith("/"):
            post_endpoint = f"{decoded_params['proxy_url']}user/auth"
        else:
            post_endpoint = f"{decoded_params['proxy_url']}/user/auth"

        try:
            assert email.split("@")[1] in decoded_params["accepted_email_subdomain"]
        except:
            raise Exception(
                f"Only emails from {decoded_params['accepted_email_subdomain']} are allowed"
            )
        response = requests.post(
            post_endpoint, json={"user_email": email, "page": page_param}
        )

        if response.status_code == 200:
            # Success!
            st.success(f"Email sent successfully!")


def user_page(page_param: str, user_id: str, token: str):
    st.title("User Configuration")

    # When the button is clicked
    if st.button("Create Key"):
        b64_flag, decoded_params = is_base64(sb=page_param)
        # Define the endpoint you want to make a POST request to
        if decoded_params["proxy_url"].endswith("/"):
            post_endpoint = f"{decoded_params['proxy_url']}key/generate"
        else:
            post_endpoint = f"{decoded_params['proxy_url']}/key/generate"
        # Make a POST request to the endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            post_endpoint, json={"duration": "1hr", "user_id": user_id}, headers=headers
        )

        if response.status_code == 200:
            # Success! You can handle the JSON response if you're expecting one
            st.success("Key created successfully!")
            response_data = response.json()
            st.success(f"API Key: {response_data['key']}")


def router():
    query_params = st.experimental_get_query_params()
    page_param = query_params.get("page", None)
    if (
        query_params.get("token", None) is not None
        and query_params.get("user_id", None) is not None
        and page_param is not None
    ):
        # render user page
        user_page(
            page_param=page_param[0],
            user_id=query_params.get("user_id")[0],
            token=query_params.get("token")[0],
        )
    elif page_param is not None:
        auth_page(page_param=page_param[0])
    else:
        st.write("Please setup proxy")


router()
