# SCIM Provisioning Fix: Handling Existing Users

## Issue Description

The original issue occurred when provisioning a user that already exists:

```
Failed to create User 'robertpo@andrew.cmu.edu' in customappsso; Error: StatusCode: BadRequest Message: Processing of the HTTP request resulted in an exception. Please see the HTTP response returned by the 'Response' property of this exception for details. Web Response: {"error":{"message":"{'error': 'User with email robertpo@andrew.cmu.edu already exists'}","type":"internal_server_error","param":"None","code":"400"}}
```

## Root Cause

The previous SCIM implementation was not following the SCIM RFC 7644 specification for handling conflicts when creating resources that already exist.

**Problems with the original implementation:**

1. **Incomplete conflict checking**: Only checked for `userName` conflicts, not email conflicts
2. **Non-compliant error format**: Returned generic HTTPException instead of proper SCIM error response
3. **Missing SCIM error type**: Did not include the required `scimType` field
4. **Wrong status code handling**: Did not follow SCIM error response structure

## SCIM Specification Requirements

According to **SCIM RFC 7644, Section 3.3**:

> "If the service provider determines that the creation of the requested resource conflicts with existing resources (e.g., a "User" resource with a duplicate "userName"), the service provider MUST return HTTP status code 409 (Conflict) with a "scimType" error code of "uniqueness", as per Section 3.12."

## Solution Implemented

### 1. Added SCIM Error Response Type

Created `SCIMErrorResponse` class in `litellm/types/proxy/management_endpoints/scim_v2.py`:

```python
class SCIMErrorResponse(BaseModel):
    """
    SCIM error response according to RFC 7644 Section 3.12
    """
    schemas: List[str] = ["urn:ietf:params:scim:api:messages:2.0:Error"]
    scimType: Optional[str] = None  # Error codes like "uniqueness", "invalidFilter", etc.
    detail: Optional[str] = None    # Human-readable description
    status: str                     # HTTP status code as string
```

### 2. Enhanced User Creation Conflict Detection

Updated `create_user` function in `litellm/proxy/management_endpoints/scim/scim_v2.py`:

- **Check userName conflicts**: Verifies if a user with the same `userName` already exists
- **Check email conflicts**: Verifies if a user with the same email already exists  
- **Proper error responses**: Returns SCIM-compliant error with `scimType: "uniqueness"`

### 3. Enhanced Group Creation Conflict Detection

Updated `create_group` function to also return SCIM-compliant error responses when groups already exist.

## Expected Behavior After Fix

### When creating a user that already exists by userName:

**Request:**
```json
POST /scim/v2/Users
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
    "userName": "robertpo@andrew.cmu.edu",
    "emails": [{"value": "robertpo@andrew.cmu.edu", "primary": true}]
}
```

**Response:**
```http
HTTP/1.1 409 Conflict
Content-Type: application/scim+json

{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
    "scimType": "uniqueness",
    "detail": "User with userName 'robertpo@andrew.cmu.edu' already exists",
    "status": "409"
}
```

### When creating a user that already exists by email:

**Response:**
```http
HTTP/1.1 409 Conflict
Content-Type: application/scim+json

{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
    "scimType": "uniqueness", 
    "detail": "User with email 'robertpo@andrew.cmu.edu' already exists",
    "status": "409"
}
```

## Benefits of This Fix

1. **SCIM RFC 7644 Compliant**: Follows the official SCIM specification exactly
2. **Better Integration**: Microsoft Entra ID and other SCIM clients will properly handle these responses
3. **Improved Error Handling**: Clear, structured error messages that SCIM clients can parse
4. **Comprehensive Conflict Detection**: Catches both userName and email conflicts
5. **Consistent Error Format**: All conflict errors now follow the same SCIM structure

## Files Modified

1. `litellm/types/proxy/management_endpoints/scim_v2.py` - Added SCIMErrorResponse type
2. `litellm/proxy/management_endpoints/scim/scim_v2.py` - Updated create_user and create_group functions

## Testing

The existing SCIM test in `tests/scim_tests/scim_e2e_test.json` expects a 409 status code when recreating users with same values, which this fix maintains while improving the response format to be SCIM-compliant.