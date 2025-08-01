"""
Google Cloud IAM authentication for Redis connections
"""
import time
from typing import Optional, Callable, Any
from redis.backoff import ConstantBackoff
from redis.retry import Retry
from redis.exceptions import (
    ConnectionError,
    AuthenticationWrongNumberOfArgsError,
    AuthenticationError
)
from redis.utils import str_if_bytes

try:
    from google.cloud import iam_credentials_v1
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    iam_credentials_v1 = None  # type: ignore


class GoogleCloudIAMAuth:
    """
    Google Cloud IAM authentication for Redis connections
    """
    
    def __init__(self, service_account: str, token_cache_ttl: float = 3500.0):
        """
        Initialize Google Cloud IAM authentication
        
        Args:
            service_account: The service account to use for authentication
            token_cache_ttl: Token cache TTL in seconds (default: 3500)
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError(
                "google-cloud-iam package is required for IAM authentication. "
                "Install it with: pip install google-cloud-iam"
            )
        
        self.service_account = service_account
        self._token_cache: Optional[str] = None
        self._token_expires_at = 0
        self._token_cache_ttl = token_cache_ttl
        
    def generate_access_token(self) -> str:
        """
        Generate an access token using Google Cloud IAM
        """
        if not GOOGLE_CLOUD_AVAILABLE or iam_credentials_v1 is None:
            raise ImportError("google-cloud-iam is not available")
            
        # Check if we have a cached token that's still valid
        if self._token_cache and time.time() < self._token_expires_at:
            return self._token_cache
            
        # Create a client
        client = iam_credentials_v1.IAMCredentialsClient()

        # Initialize request argument(s)
        request = iam_credentials_v1.GenerateAccessTokenRequest(
            name=self.service_account,
            scope=['https://www.googleapis.com/auth/cloud-platform'],
        )

        # Make the request
        response = client.generate_access_token(request=request)

        # Cache the token
        self._token_cache = str(response.access_token)
        self._token_expires_at = time.time() + self._token_cache_ttl
        
        return self._token_cache


def create_iam_redis_connect_func(service_account: str) -> Callable:
    """
    Create a Redis connect function that uses Google Cloud IAM authentication
    
    Args:
        service_account: The service account to use for authentication
        
    Returns:
        A function that can be used as redis_connect_func
    """
    auth = GoogleCloudIAMAuth(service_account)
    
    def iam_connect(connection):
        """Initialize the connection and authenticate using IAM"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("IAM authentication timeout")
        
        try:
            connection._parser.on_connect(connection)

            # Set a timeout for the entire authentication process
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout for sync auth
            
            try:
                auth_args = (auth.generate_access_token(),)
                connection.send_command("AUTH", *auth_args, check_health=False)

                try:
                    auth_response = connection.read_response()
                except AuthenticationWrongNumberOfArgsError:
                    connection.send_command("AUTH", connection.password, check_health=False)
                    auth_response = connection.read_response()

                if str_if_bytes(auth_response) != "OK":
                    raise AuthenticationError("Invalid Username or Password")
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)
                
        except TimeoutError:
            raise AuthenticationError("Timeout during IAM authentication")
        except Exception as e:
            raise AuthenticationError(f"IAM authentication failed: {str(e)}")
    
    return iam_connect


def create_async_iam_redis_connect_func(service_account: str) -> Callable:
    """
    Create an async Redis connect function that uses Google Cloud IAM authentication
    
    Args:
        service_account: The service account to use for authentication
        
    Returns:
        A function that can be used as redis_connect_func for async Redis
    """
    auth = GoogleCloudIAMAuth(service_account)
    
    async def async_iam_connect(connection):
        """Initialize the async connection and authenticate using IAM"""
        import asyncio
        
        try:
            connection._parser.on_connect(connection)

            # Generate access token with timeout
            token = await asyncio.wait_for(
                asyncio.to_thread(auth.generate_access_token),
                timeout=15.0  # Increased timeout for token generation
            )
            
            auth_args = (token,)
            await connection.send_command("AUTH", *auth_args, check_health=False)

            try:
                # Use a longer timeout for the auth response
                auth_response = await asyncio.wait_for(
                    connection.read_response(),
                    timeout=20.0  # 20 second timeout for auth response
                )
            except asyncio.TimeoutError:
                raise AuthenticationError("Timeout waiting for Redis AUTH response")
            except AuthenticationWrongNumberOfArgsError:
                await connection.send_command("AUTH", connection.password, check_health=False)
                auth_response = await asyncio.wait_for(
                    connection.read_response(),
                    timeout=20.0
                )

            if str_if_bytes(auth_response) != "OK":
                raise AuthenticationError("Invalid Username or Password")
                
        except asyncio.TimeoutError as e:
            raise AuthenticationError(f"Timeout during IAM authentication: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"IAM authentication failed: {str(e)}")
    
    return async_iam_connect 