import pytest
import requests
import time
from typing import Dict, List
import logging
from litellm._uuid import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamAPI:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def create_team(self, team_alias: str, models: List[str] = None) -> Dict:
        """Create a new team"""
        # Generate a unique team_id using uuid
        team_id = f"test_team_{uuid.uuid4().hex[:8]}"

        data = {
            "team_id": team_id,
            "team_alias": team_alias,
            "models": models or ["o3-mini"],
        }

        response = requests.post(
            f"{self.base_url}/team/new", headers=self.headers, json=data
        )
        response.raise_for_status()
        logger.info(f"Created new team: {team_id}")
        return response.json(), team_id

    def get_team_info(self, team_id: str) -> Dict:
        """Get current team information"""
        response = requests.get(
            f"{self.base_url}/team/info",
            headers=self.headers,
            params={"team_id": team_id},
        )
        response.raise_for_status()
        return response.json()

    def add_team_member(self, team_id: str, user_email: str, role: str) -> Dict:
        """Add a single team member"""
        data = {"team_id": team_id, "member": [{"role": role, "user_id": user_email}]}
        response = requests.post(
            f"{self.base_url}/team/member_add", headers=self.headers, json=data
        )
        response.raise_for_status()
        return response.json()

    def delete_team_member(self, team_id: str, user_id: str) -> Dict:
        """Delete a team member

        Args:
            team_id (str): ID of the team
            user_id (str): User ID to remove from team

        Returns:
            Dict: Response from the API
        """
        data = {"team_id": team_id, "user_id": user_id}
        response = requests.post(
            f"{self.base_url}/team/member_delete", headers=self.headers, json=data
        )
        response.raise_for_status()
        return response.json()


@pytest.fixture
def api_client():
    """Fixture for TeamAPI client"""
    base_url = "http://localhost:4000"
    auth_token = "sk-1234"  # Replace with your token
    return TeamAPI(base_url, auth_token)


@pytest.fixture
def new_team(api_client):
    """Fixture that creates a new team for each test"""
    team_alias = f"Test Team {uuid.uuid4().hex[:6]}"
    team_response, team_id = api_client.create_team(team_alias)
    logger.info(f"Created test team: {team_id} ({team_alias})")
    return team_id


def verify_member_in_team(team_info: Dict, user_email: str) -> bool:
    """Verify if a member exists in team"""
    return any(
        member["user_id"] == user_email
        for member in team_info["team_info"]["members_with_roles"]
    )


def test_team_creation(api_client):
    """Test team creation"""
    team_alias = f"Test Team {uuid.uuid4().hex[:6]}"
    team_response, team_id = api_client.create_team(team_alias)

    # Verify team was created
    team_info = api_client.get_team_info(team_id)
    assert team_info["team_id"] == team_id
    assert team_info["team_info"]["team_alias"] == team_alias
    assert "o3-mini" in team_info["team_info"]["models"]


def test_add_single_member(api_client, new_team):
    """Test adding a single member to a new team"""
    # Get initial team info
    initial_info = api_client.get_team_info(new_team)
    initial_size = len(initial_info["team_info"]["members_with_roles"])

    # Add new member
    test_email = f"pytest_user_{uuid.uuid4().hex[:6]}@mycompany.com"
    api_client.add_team_member(new_team, test_email, "user")

    # Allow time for system to process
    time.sleep(1)

    # Verify addition
    updated_info = api_client.get_team_info(new_team)
    updated_size = len(updated_info["team_info"]["members_with_roles"])

    # Assertions
    assert verify_member_in_team(
        updated_info, test_email
    ), f"Member {test_email} not found in team"
    assert (
        updated_size == initial_size + 1
    ), f"Team size did not increase by 1 (was {initial_size}, now {updated_size})"


def test_add_multiple_members(api_client, new_team):
    """Test adding multiple members to a new team"""
    # Get initial team size
    initial_info = api_client.get_team_info(new_team)
    initial_size = len(initial_info["team_info"]["members_with_roles"])

    # Add 10 members
    added_emails = []
    for i in range(10):
        email = f"pytest_user_{uuid.uuid4().hex[:6]}@mycompany.com"
        added_emails.append(email)

        logger.info(f"Adding member {i+1}/10: {email}")
        api_client.add_team_member(new_team, email, "user")

        # Allow time for system to process
        time.sleep(1)

        # Verify after each addition
        current_info = api_client.get_team_info(new_team)
        current_size = len(current_info["team_info"]["members_with_roles"])

        # Assertions for each addition
        assert verify_member_in_team(
            current_info, email
        ), f"Member {email} not found in team"
        assert (
            current_size == initial_size + i + 1
        ), f"Team size incorrect after adding {email}"

    # Final verification
    final_info = api_client.get_team_info(new_team)
    final_size = len(final_info["team_info"]["members_with_roles"])

    # Final assertions
    assert (
        final_size == initial_size + 10
    ), f"Final team size incorrect (expected {initial_size + 10}, got {final_size})"
    for email in added_emails:
        assert verify_member_in_team(
            final_info, email
        ), f"Member {email} not found in final team check"


def test_team_info_structure(api_client, new_team):
    """Test the structure of team info response"""
    team_info = api_client.get_team_info(new_team)

    # Verify required fields exist
    assert "team_id" in team_info
    assert "team_info" in team_info
    assert "members_with_roles" in team_info["team_info"]
    assert "models" in team_info["team_info"]

    # Verify member structure
    if team_info["team_info"]["members_with_roles"]:
        member = team_info["team_info"]["members_with_roles"][0]
        assert "user_id" in member
        assert "role" in member


def test_error_handling(api_client):
    """Test error handling for invalid team ID"""
    with pytest.raises(requests.exceptions.HTTPError):
        api_client.get_team_info("invalid-team-id")


def test_duplicate_user_addition(api_client, new_team):
    """Test that adding the same user twice is handled appropriately"""
    # Add user first time
    test_email = f"pytest_user_{uuid.uuid4().hex[:6]}@mycompany.com"
    initial_response = api_client.add_team_member(new_team, test_email, "user")

    # Allow time for system to process
    time.sleep(1)

    # Get team info after first addition
    team_info_after_first = api_client.get_team_info(new_team)
    size_after_first = len(team_info_after_first["team_info"]["members_with_roles"])

    logger.info(f"First addition completed. Team size: {size_after_first}")

    # Attempt to add same user again
    with pytest.raises(requests.exceptions.HTTPError):
        api_client.add_team_member(new_team, test_email, "user")

    # Allow time for system to process
    time.sleep(1)

    # Get team info after second addition attempt
    team_info_after_second = api_client.get_team_info(new_team)
    size_after_second = len(team_info_after_second["team_info"]["members_with_roles"])

    # Verify team size didn't change
    assert (
        size_after_second == size_after_first
    ), f"Team size changed after duplicate addition (was {size_after_first}, now {size_after_second})"

    # Verify user appears exactly once
    user_count = sum(
        1
        for member in team_info_after_second["team_info"]["members_with_roles"]
        if member["user_id"] == test_email
    )
    assert user_count == 1, f"User appears {user_count} times in team (expected 1)"

    logger.info(f"Duplicate addition attempted. Final team size: {size_after_second}")
    logger.info(f"Number of times user appears in team: {user_count}")


def test_member_deletion(api_client, new_team):
    """Test that member deletion works correctly and removes all instances of a user"""
    # Add a test user
    user_id = f"pytest_user_{uuid.uuid4().hex[:6]}"
    api_client.add_team_member(new_team, user_id, "user")
    time.sleep(1)

    # Verify user was added
    team_info_before = api_client.get_team_info(new_team)
    assert verify_member_in_team(
        team_info_before, user_id
    ), "User was not added successfully"

    initial_size = len(team_info_before["team_info"]["members_with_roles"])

    # Attempt to delete the same user multiple times (5 times)
    for i in range(5):
        logger.info(f"Attempting deletion {i+1}/5")
        if i == 0:
            # First deletion should succeed
            api_client.delete_team_member(new_team, user_id)
            time.sleep(1)
        else:
            # Subsequent deletions should raise an error
            try:
                api_client.delete_team_member(new_team, user_id)
                pytest.fail("Expected HTTPError for duplicate deletion")
            except requests.exceptions.HTTPError as e:
                logger.info(
                    f"Expected error received on deletion attempt {i+1}: {str(e)}"
                )

    # Verify final state
    final_info = api_client.get_team_info(new_team)
    final_size = len(final_info["team_info"]["members_with_roles"])

    # Verify user is completely removed
    assert not verify_member_in_team(
        final_info, user_id
    ), "User still exists in team after deletion"

    # Verify only one member was removed
    assert (
        final_size == initial_size - 1
    ), f"Team size changed unexpectedly (was {initial_size}, now {final_size})"


def test_delete_nonexistent_member(api_client, new_team):
    """Test that attempting to delete a nonexistent member raises appropriate error"""
    nonexistent_user = f"nonexistent_{uuid.uuid4().hex[:6]}"

    # Verify user doesn't exist first
    team_info = api_client.get_team_info(new_team)
    assert not verify_member_in_team(
        team_info, nonexistent_user
    ), "Test setup error: nonexistent user somehow exists"

    # Attempt to delete nonexistent user
    try:
        api_client.delete_team_member(new_team, nonexistent_user)
        pytest.fail("Expected HTTPError for deleting nonexistent user")
    except requests.exceptions.HTTPError as e:
        logger.info(f"Expected error received: {str(e)}")
        assert e.response.status_code == 400
