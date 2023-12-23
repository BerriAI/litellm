"""
Auth in user, to proxy ui. 

Uses supabase passwordless auth: https://supabase.com/docs/reference/python/auth-signinwithotp

Remember to set your redirect url to 8501 (streamlit default).
"""
import logging
logging.basicConfig(level=logging.DEBUG)
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client, Client

# Set up Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def sign_in_with_otp(email: str, redirect_url: str):
    data = supabase.auth.sign_in_with_otp({"email": email,
                                           "options": {
                                               "email_redirect_to": redirect_url
                                               }})
    print(f"data: {data}")
    # Redirect to Supabase UI with the return data
    st.write(f"Please check your email for a login link!")
    

# Create the Streamlit app
def auth_page(redirect_url: str):
    st.title("User Authentication")

    # User email input
    email = st.text_input("Enter your email")

    # Sign in button
    if st.button("Sign In"):
        sign_in_with_otp(email, redirect_url=redirect_url)
