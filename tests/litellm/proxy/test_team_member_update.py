import requests

def test_non_premium_user_cannot_promote_to_admin():
    """
    Non-premium users should not be able to promote a team member to admin.
    """
    # Setup: Use your actual test server and test data
    base_url = "http://localhost:4000"
    access_token = "sk-1234"  # Replace with a real non-premium token
    team_id = "test-team-id"         # Replace with a real team ID
    user_id = "test-user-id"         # Replace with a real user ID

    url = f"{base_url}/team/member_update"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "team_id": team_id,
        "user_id": user_id,
        "role": "admin"
    }

    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
