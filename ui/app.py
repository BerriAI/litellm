import streamlit as st
import urllib.parse

def generate_key(name, description):
    # Code to generate and return a key goes here
    return "Generated Key"


def main(api_base, url_hash):
    st.title("Key Request")

    # Create input fields for key name and description
    name = st.text_input("Key Name")
    description = st.text_area("Key Description")

    # Create a button to request the key
    if st.button("Request Key"):
        if name and description:
            key = generate_key(name, description)
            st.success(f"Your key: {key}")
        else:
            st.error("Please enter a valid key name and description.")


if __name__ == "__main__":
    # Get the proxy URL and hash from the admin
    proxy_url = st.text_input("Admin Proxy URL")
    hash_input = st.text_input("URL Hash")

    # Generate the public URL with hash
    encoded_hash = urllib.parse.quote(hash_input.strip()) if hash_input else ""
    public_url = f"{proxy_url}/{encoded_hash}"

    # Run the Streamlit app
    main(proxy_url, hash_input)