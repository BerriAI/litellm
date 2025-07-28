import os
from litellm import completion

oci_region = os.environ.get("OCI_REGION")
oci_user = os.environ.get("OCI_USER")
oci_fingerprint = os.environ.get("OCI_FINGERPRINT")
oci_tenancy = os.environ.get("OCI_TENANCY")
oci_key = os.environ.get("OCI_KEY")
oci_compartment_id = os.environ.get("OCI_COMPARTMENT_ID")

response = completion(
    model="oci/meta.llama-3.3-70b-instruct",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What is the capital of France?"}],
    oci_region=oci_region,
    oci_user=oci_user,
    oci_fingerprint=oci_fingerprint,
    oci_tenancy=oci_tenancy,
    oci_key=oci_key,
    oci_compartment_id=oci_compartment_id,
)

print(response)
