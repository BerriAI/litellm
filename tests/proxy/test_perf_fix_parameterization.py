
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# MOCK PRISMA MODULE BEFORE IMPORT
mock_prisma_module = MagicMock()
sys.modules["prisma"] = mock_prisma_module
mock_prisma_module.Prisma = MagicMock() # Mock the Prisma class

mock_prisma_errors = MagicMock()
# Create a dummy exception class for PrismaError
class MockPrismaError(Exception): pass
mock_prisma_errors.PrismaError = MockPrismaError
sys.modules["prisma.errors"] = mock_prisma_errors

try:
    from litellm.proxy.utils import PrismaClient
except ImportError:
    # Handle cases where litellm might not be directly importable without setup
    print("Could not import litellm.proxy.utils.PrismaClient directly.")
    sys.exit(1)

class TestPerfFixParameterization(unittest.IsolatedAsyncioTestCase):
    async def test_get_data_parameterization(self):
        """
        Verify that PrismaClient.get_data uses parameterized queries for the combined_view.
        """
        # 1. Setup Mock
        mock_proxy_logging = MagicMock()
        mock_proxy_logging.failure_handler = AsyncMock()
        
        # Mock the PrismaWrapper and its query_first method
        mock_db = MagicMock()
        mock_response = {
            "token": "hashed-test-key", 
            "user_id": "test-user",
            "team_models": [],
            "team_blocked": False,
            "team_members_with_roles": None,
            "expires": None,
            "key_alias": "test-key"
        }
        mock_db.query_first = AsyncMock(return_value=mock_response)
        
        # Create client with mocked internals
        # We might need to mock os.getenv to avoid side effects during init if any
        client = PrismaClient(database_url="postgresql://test", proxy_logging_obj=mock_proxy_logging)
        client.db = mock_db # Inject mock db
        
        # 2. Execute get_data
        test_token = "test-key-123"
        # We need to ensure we hit the combined_view logic
        await client.get_data(token=test_token, table_name="combined_view")
        
        # 3. Verify
        # Check arguments passed to query_first
        call_args = mock_db.query_first.call_args
        self.assertIsNotNone(call_args, "query_first was not called")
        
        # call_args[0] contains positional args
        sql_query = call_args[0][0]
        query_params = call_args[0][1:]
        
        # Check for parameter placeholder
        print(f"Executed SQL: {sql_query}")
        self.assertIn("$1", sql_query, "SQL query should contain '$1' for parameterization")
        self.assertNotIn("'{token}'", sql_query, "SQL query should NOT contain f-string interpolated token")
        self.assertIn("v.token = $1", sql_query, "SQL query should use 'v.token = $1'")
        
        # Check that the token was passed as an argument
        self.assertTrue(len(query_params) > 0, "No parameters passed to query_first")
        
        actual_param = query_params[0]
        print(f"Actual param passed: {actual_param}")
        self.assertIsNotNone(actual_param)

if __name__ == "__main__":
    unittest.main()
