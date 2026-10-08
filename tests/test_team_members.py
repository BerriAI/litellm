import os
import pytest
import requests
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
    auth_token = os.environ["LITELLM_MASTER_KEY"]  # Replace with your token
    return TeamAPI(base_url, auth_token)


def test_team_creation(api_client):
    """Test team creation"""
    team_alias = f"Test Team {uuid.uuid4().hex[:6]}"
    team_response, team_id = api_client.create_team(team_alias)

    # Verify team was created
    team_info = api_client.get_team_info(team_id)
    assert team_info["team_id"] == team_id
    assert team_info["team_info"]["team_alias"] == team_alias
    assert "o3-mini" in team_info["team_info"]["models"]


