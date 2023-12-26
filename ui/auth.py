"""
Auth in user, to proxy ui.

Uses supabase passwordless auth: https://supabase.com/docs/reference/python/auth-signinwithotp

Remember to set your redirect url to 8501 (streamlit default).
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os
from supabase import create_client, Client

# Set up Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def sign_in_with_otp(email: str, page_param: str):
    print(f"received page param: {page_param}")
    data = supabase.auth.sign_in_with_otp(
        {"email": email, "options": {"data": {"page_param": page_param}}}
    )
    print(f"data: {data}")
    # Redirect to Supabase UI with the return data
    st.write(f"Please check your email for a login link!")


def verify_with_otp(token: str):
    res = supabase.auth.verify_otp({"token_hash": token, "type": "email"})
    return res


# Create the Streamlit app
def auth_page(page_param: str):
    st.title("User Authentication")

    # User email input
    email = st.text_input("Enter your email")

    # Sign in button
    if st.button("Sign In"):
        sign_in_with_otp(email, page_param=page_param)
