import sys
import os
import time
import jwt
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litellm.proxy.auth.handle_jwt import JWTHandler, JWTAuthManager
from litellm.proxy._types import LiteLLM_JWTAuth, LitellmUserRoles, LiteLLM_UserTable


@pytest.fixture
def jwt_setup():
    # Create a JWT token for testing
    test_jwt_secret = "test_secret"
    test_user_id = "test_user_123"
    test_email = "test@example.com"
    
    token_payload = {
        "sub": test_user_id,
        "email": test_email,
        "exp": int(time.time()) + 3600  # 1 hour from now
    }
    test_token = jwt.encode(
        token_payload, 
        test_jwt_secret, 
        algorithm="HS256"
    )
    
    # Create handler and simple mocks
    jwt_handler = JWTHandler()
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    
    # Patch the auth_jwt method directly
    with patch.object(JWTHandler, 'auth_jwt', return_value=token_payload):
        yield {
            'jwt_handler': jwt_handler,
            'prisma_client': mock_prisma_client,
            'cache': mock_cache,
            'token': test_token,
            'token_payload': token_payload,
            'user_id': test_user_id,
            'email': test_email
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_role,user_id_upsert,expected_role",
    [
        (LitellmUserRoles.INTERNAL_USER, True, LitellmUserRoles.INTERNAL_USER),  # Default role (explicit)
        (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, True, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),  # Custom role
        (LitellmUserRoles.ORG_ADMIN, True, LitellmUserRoles.ORG_ADMIN),  # Admin role
        (LitellmUserRoles.INTERNAL_USER, False, LitellmUserRoles.INTERNAL_USER),  # No upsert, but role still passed
    ]
)
async def test_jwt_user_role_assignment(jwt_setup, default_role, user_id_upsert, expected_role):
    """
    Test that the correct role is assigned to new users created via JWT authentication
    based on the default_user_role configuration.
    """
    
    default_role_capture = {}
    
    # Create a mock user object to return after "creation"
    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_id = jwt_setup['user_id']
    mock_user.user_role = expected_role
    
    # Mock all the get_* functions
    async def mock_get_user_object(user_id, prisma_client, user_api_key_cache, user_id_upsert, 
                           parent_otel_span=None, proxy_logging_obj=None, sso_user_id=None, 
                           user_email=None, check_db_only=None, default_user_role=None):
        default_role_capture['role'] = default_user_role
        default_role_capture['upsert'] = user_id_upsert
        
        if not user_id_upsert:
            raise ValueError("User not found")
        return mock_user
    
    async def mock_get_org_object(*args, **kwargs):
        return None
    
    async def mock_get_end_user_object(*args, **kwargs):
        return None
    
    # Apply the patches
    with patch('litellm.proxy.auth.handle_jwt.get_user_object', mock_get_user_object), \
         patch('litellm.proxy.auth.handle_jwt.get_org_object', mock_get_org_object), \
         patch('litellm.proxy.auth.handle_jwt.get_end_user_object', mock_get_end_user_object):
        
        # Configure JWT handler
        jwt_auth_config = LiteLLM_JWTAuth(
            user_id_jwt_field="sub",
            user_email_jwt_field="email",
            user_id_upsert=user_id_upsert,
            default_user_role=default_role
        )
        jwt_setup['jwt_handler'].update_environment(
            prisma_client=jwt_setup['prisma_client'],
            user_api_key_cache=jwt_setup['cache'],
            litellm_jwtauth=jwt_auth_config
        )
        
        # Get user info and objects
        jwt_valid_token = await jwt_setup['jwt_handler'].auth_jwt(token=jwt_setup['token'])
        user_id, user_email, valid_user_email = await JWTAuthManager.get_user_info(
            jwt_handler=jwt_setup['jwt_handler'],
            jwt_valid_token=jwt_valid_token
        )
        
        if not user_id_upsert:
            # This should raise an exception since upsert is False and user doesn't exist
            with pytest.raises(ValueError, match="User not found"):
                await JWTAuthManager.get_objects(
                    user_id=user_id,
                    user_email=user_email,
                    org_id=None,
                    end_user_id=None,
                    valid_user_email=valid_user_email,
                    jwt_handler=jwt_setup['jwt_handler'],
                    prisma_client=jwt_setup['prisma_client'],
                    user_api_key_cache=jwt_setup['cache'],
                    parent_otel_span=None,
                    proxy_logging_obj=MagicMock()
                )
            # Verify upsert setting was passed correctly
            assert default_role_capture['upsert'] is False
        else:
            # Get objects should succeed
            user_obj, org_obj, end_user_obj = await JWTAuthManager.get_objects(
                user_id=user_id,
                user_email=user_email,
                org_id=None,
                end_user_id=None,
                valid_user_email=valid_user_email,
                jwt_handler=jwt_setup['jwt_handler'],
                prisma_client=jwt_setup['prisma_client'],
                user_api_key_cache=jwt_setup['cache'],
                parent_otel_span=None,
                proxy_logging_obj=MagicMock()
            )
            
            # Verify role was passed correctly
            assert default_role_capture['role'] == expected_role
            assert user_obj == mock_user
            assert org_obj is None
            assert end_user_obj is None


@pytest.mark.asyncio
async def test_jwt_config_options():
    """Test various JWT configuration options and their effect on default roles"""
    
    # Test that default_user_role can be explicitly set
    config = LiteLLM_JWTAuth(
        user_id_jwt_field="sub",
        default_user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    )
    assert config.default_user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    
    # Test default value when not specified
    config = LiteLLM_JWTAuth(user_id_jwt_field="sub")
    assert config.default_user_role == LitellmUserRoles.INTERNAL_USER
    
    # Test with org_admin role
    config = LiteLLM_JWTAuth(
        user_id_jwt_field="sub",
        default_user_role=LitellmUserRoles.ORG_ADMIN
    )
    assert config.default_user_role == LitellmUserRoles.ORG_ADMIN 