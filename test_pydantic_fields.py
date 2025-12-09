from litellm.proxy._types import GenerateKeyRequest

# Test 1: Check if fields exist in model
print("=== Test 1: Check model_fields ===")
print(
    f"rpm_limit_type in model_fields: {'rpm_limit_type' in GenerateKeyRequest.model_fields}"
)
print(
    f"tpm_limit_type in model_fields: {'tpm_limit_type' in GenerateKeyRequest.model_fields}"
)

# Test 2: Create instance with empty dict (simulating FastAPI parsing minimal request)
print("\n=== Test 2: Create instance with minimal data ===")
instance = GenerateKeyRequest()
print(f"Instance created: {instance}")
print(f"Instance dict: {instance.model_dump()}")

# Test 3: Try to access the fields
print("\n=== Test 3: Try direct attribute access ===")
try:
    print(f"instance.rpm_limit_type = {instance.rpm_limit_type}")
    print(f"instance.tpm_limit_type = {instance.tpm_limit_type}")
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test 4: Try getattr
print("\n=== Test 4: Try getattr ===")
print(
    f"getattr(instance, 'rpm_limit_type', None) = {getattr(instance, 'rpm_limit_type', None)}"
)
print(
    f"getattr(instance, 'tpm_limit_type', None) = {getattr(instance, 'tpm_limit_type', None)}"
)

# Test 5: Check what fields are actually set
print("\n=== Test 5: Check model_fields_set ===")
print(f"model_fields_set: {instance.model_fields_set}")

# Test 6: Check instance __dict__
print("\n=== Test 6: Check instance __dict__ ===")
print(f"'rpm_limit_type' in __dict__: {'rpm_limit_type' in instance.__dict__}")
print(f"'tpm_limit_type' in __dict__: {'tpm_limit_type' in instance.__dict__}")
