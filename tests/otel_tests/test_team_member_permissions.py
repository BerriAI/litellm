"""
1. Default permissions for members in a team - allowed to call /key/info and /key/health
    - Create a team, create a member in a team (role = "user")


    Invalid Permissions:
    - User tries creating a key with team_id = team_id -> expect to fail. Invalid Permissions 
    - User tries editing a key with team_id = team_id -> expect to fail. Invalid Permissions 
    - User tries deleting a key with team_id = team_id -> expect to fail. Invalid Permissions 
    - User tries regenerating a key with team_id = team_id -> expect to fail. Invalid Permissions 

    Valid Permissions:
    - User tries calling /key/info with team_id, expect to get valid response 



2. Permissions - members allowd to edit, delete keys but not allowed to create keys
    - Create a team with member_permissions = ["/key/update", "/key/delete", "/key/info"]
    - Create a member in the team with role = "user"

    Valid Permissions:
    - User tries editing a key with team_id = team_id -> expect to pass. Valid Permissions
    - User tries deleting a key with team_id = team_id -> expect to pass. Valid Permissions


    - User tries creating a key with team_id = team_id -> expect to fail. Invalid Permissions
    - User tries regenerating a key with team_id = team_id -> expect to fail. Invalid Permissions
    - User tries calling /key/info with team_id, expect to get valid response 



3. Permissions - members allowed to create keys but not allowed to edit, delete keys
    - Create a team with member_permissions = ["/key/generate"]
    - Create a member in the team with role = "user"

    Valid Permissions:
    - User tries creating a key with team_id = team_id -> expect to pass. Valid Permissions

    Invalid Permissions:
    - User tries editing a key with team_id = team_id -> expect to fail. Invalid Permissions
    - User tries deleting a key with team_id = team_id -> expect to fail. Invalid Permissions
    - User tries regenerating a key with team_id = team_id -> expect to fail. Invalid Permissions
"""

import pytest
import asyncio
import aiohttp, openai
from litellm._uuid import uuid
import json
from litellm.proxy._types import ProxyErrorTypes
from typing import Optional
LITELLM_MASTER_KEY = "sk-1234"

async def create_team(session, key, member_permissions=None):
    url = "http://0.0.0.0:4000/team/new"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "team_member_permissions": member_permissions
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()

async def create_user(session, key, user_id, team_id=None):
    url = "http://0.0.0.0:4000/user/new"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "user_id": user_id
    }
    if team_id:
        data["team_id"] = team_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()

async def add_team_member(session, key, team_id, user_id, role="user"):
    url = "http://0.0.0.0:4000/team/member_add"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "team_id": team_id,
        "member": {
            "role": role,
            "user_id": user_id
        }
    }
    print("Adding team member with data: ", data)
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()

async def generate_key(session, key, team_id=None, user_id=None):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {}
    if team_id:
        data["team_id"] = team_id
    if user_id:
        data["user_id"] = user_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        
        if status != 200:
            return {"status": status, "error": response_text}

        return await response.json()

async def key_info(session, key, key_id):
    url = f"http://0.0.0.0:4000/key/info?key={key_id}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        
        if status != 200:
            return {"status": status, "error": response_text}

        return await response.json()

async def update_key(
    session: aiohttp.ClientSession,
    key: str,
    key_id: str,
    team_id: Optional[str] = None,
):
    """
    Update a key

    Args:
        key: key to use for authentication
        key_id: key to update
    """
    url = "http://0.0.0.0:4000/key/update"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "key": key_id,
        "metadata": {"updated": True}
    }
    if team_id:
        data["team_id"] = team_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        
        if status != 200:
            return {"status": status, "error": response_text}

        return await response.json()

async def delete_key(session, key, key_id):
    url = "http://0.0.0.0:4000/key/delete"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "keys": [key_id]
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        
        if status != 200:
            return {"status": status, "error": response_text}

        return await response.json()

async def regenerate_key(session, key, key_id, team_id=None):
    url = "http://0.0.0.0:4000/key/regenerate"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "key": key_id
    }
    if team_id:
        data["team_id"] = team_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        
        if status != 200:
            return {"status": status, "error": response_text}

        return await response.json()

@pytest.mark.asyncio()
async def test_default_member_permissions():
    """
    Test default permissions for members in a team - allowed to call /key/info and /key/health
    """
    async with aiohttp.ClientSession() as session:
        master_key = LITELLM_MASTER_KEY
        
        # Create a team
        team_data = await create_team(
            session=session,
            key=master_key
        )
        team_id = team_data["team_id"]

        # create a team key
        team_key_data = await generate_key(
            session=session,
            key=master_key,
            team_id=team_id
        )
        team_key = team_key_data["key"]

        # create a user
        user_data = await create_user(
            session=session,
            key=master_key,
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            team_id=team_id
        )
        user_id = user_data["user_id"]
        
        # Create a user key
        print("New user data: ", user_data)

        # Create a user key
        user_key_data = await generate_key(
            session=session,
            key=master_key,
            user_id=user_id
        )
        print("new user key: ", user_key_data)
        user_key = user_key_data["key"]
        
        # Test invalid permissions
        # User tries creating a key with team_id
        print("Regular team member trying to create a key with team_id. Expecting error.")
        create_result = await generate_key(
            session=session,
            key=user_key,
            team_id=team_id
        )
        print("result: ", create_result)
        assert "status" in create_result and create_result["status"] == 401, "User should not be able to create keys for team"
        error_data = json.loads(create_result["error"])
        print("error response =", json.dumps(error_data, indent=4))
        assert error_data["error"]["type"] == ProxyErrorTypes.team_member_permission_error.value, "Error should be a team member permission error"
        
        # User tries editing a key with team_id
        print("Regular team member trying to edit a key with team_id. Expecting error.")
        update_result = await update_key(
            session=session,
            key=user_key,
            key_id=team_key,
            team_id="ATTACKER_TEAM_ID"
        )
        assert "status" in update_result and update_result["status"] == 401, "User should not be able to update keys for team"
        error_data = json.loads(update_result["error"])
        print("error response =", json.dumps(error_data, indent=4))
        assert error_data["error"]["type"] == ProxyErrorTypes.team_member_permission_error.value, "Error should be a team member permission error"
        
        # User tries deleting a key with team_id
        print("Regular team member trying to delete a key with team_id. Expecting error.")
        delete_result = await delete_key(
            session=session,
            key=user_key,
            key_id=team_key,
        )
        assert "status" in delete_result and delete_result["status"] == 401, "User should not be able to delete keys for team"
        error_data = json.loads(delete_result["error"])
        print("error response =", json.dumps(error_data, indent=4))
        assert error_data["error"]["type"] == ProxyErrorTypes.team_member_permission_error.value, "Error should be a team member permission error"
        
        # User tries regenerating a key with team_id
        print("Regular team member trying to regenerate a key with team_id. Expecting error.")
        regenerate_result = await regenerate_key(
            session=session,
            key=user_key,
            key_id=team_key,
        )
        assert "status" in regenerate_result and regenerate_result["status"] == 401, "User should not be able to regenerate keys for team"
        error_data = json.loads(regenerate_result["error"])
        print("error response =", json.dumps(error_data, indent=4))
        assert error_data["error"]["type"] == ProxyErrorTypes.team_member_permission_error.value, "Error should be a team member permission error"
        
        # Test valid permissions
        # User tries calling /key/info with team_id
        print("Regular team member trying to get key info with team_id. Expecting success.")
        info_result = await key_info(
            session=session,
            key=user_key,
            key_id=team_key,
    )
        print("info result =", info_result)
        assert "status" not in info_result, "Admin should be able to get key info"

@pytest.mark.asyncio()
async def test_edit_delete_permissions():
    """
    Test permissions - members allowed to edit, delete keys but not allowed to create keys
    """
    async with aiohttp.ClientSession() as session:
        master_key = LITELLM_MASTER_KEY
        
        # Create a team with specific member permissions
        team_data = await create_team(
            session=session,
            key=master_key,
            member_permissions=["/key/update", "/key/delete", "/key/info"]
        )
        team_id = team_data["team_id"]
        
        # create a user in team=team_id
        user_data = await create_user(
            session=session,
            key=master_key,
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            team_id=team_id
        )
        user_id = user_data["user_id"]
        
        # Generate an admin key for the team
        admin_key_data = await generate_key(session, master_key, team_id)
        key_id = admin_key_data["key"]
        
        # Create a user key
        user_key_data = await generate_key(
            session=session,
            key=master_key,
            user_id=user_id
        )
        user_key = user_key_data["key"]
        
        # Test valid permissions
        # User tries editing a key with team_id
        update_result = await update_key(
            session=session,
            key=user_key,
            key_id=key_id,
            team_id=team_id
        )
        assert "status" not in update_result, "User should be able to update keys for team"
        
        # User tries deleting a key with team_id - test this last
        delete_result = await delete_key(
            session=session,
            key=user_key,
            key_id=key_id
        )
        assert "status" not in delete_result, "User should be able to delete keys for team"
        
        # Test invalid permissions
        # User tries creating a key with team_id
        create_result = await generate_key(
            session=session,
            key=user_key,
            team_id=team_id
        )
        assert "status" in create_result and create_result["status"] != 200, "User should not be able to create keys for team"
        
        # User tries regenerating a key with team_id
        regenerate_result = await regenerate_key(
            session=session,
            key=user_key,
            key_id=key_id,
            team_id=team_id
        )
        assert "status" in regenerate_result and regenerate_result["status"] != 200, "User should not be able to regenerate keys for team"

@pytest.mark.asyncio()
async def test_create_permissions():
    """
    Test permissions - members allowed to create keys but not allowed to edit, delete keys
    """
    async with aiohttp.ClientSession() as session:
        master_key = LITELLM_MASTER_KEY
        
        # Create a team with specific member permissions
        team_data = await create_team(
            session=session,
            key=master_key,
            member_permissions=["/key/generate"]
        )
        team_id = team_data["team_id"]
        
        # Create a user in the team
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        await add_team_member(
            session=session,
            key=master_key,
            team_id=team_id,
            user_id=user_id,
            role="user"
        )
        
        # Generate an admin key for the team
        admin_key_data = await generate_key(
            session=session,
            key=master_key,
            team_id=team_id
        )
        admin_key = admin_key_data["key"]
        key_id = admin_key_data["key"]
        
        # Create a user key
        user_key_data = await generate_key(
            session=session,
            key=master_key,
            user_id=user_id
        )
        user_key = user_key_data["key"]
        
        # Test valid permissions
        # User tries creating a key with team_id
        create_result = await generate_key(
            session=session,
            key=user_key,
            team_id=team_id
        )
        print("success, user created key for team=", create_result)
        assert "key" in create_result, "User should be able to create keys for team"
        assert create_result["team_id"] == team_id, "User should be able to create keys for team"
        assert "status" not in create_result, "User should be able to create keys for team"
        
        # Test invalid permissions
        # User tries editing a key with team_id
        update_result = await update_key(
            session=session,
            key=user_key,
            key_id=key_id,
            team_id=team_id
        )
        assert "status" in update_result and update_result["status"] != 200, "User should not be able to update keys for team"
        
        # User tries deleting a key with team_id
        delete_result = await delete_key(
            session=session,
            key=user_key,
            key_id=key_id
        )
        assert "status" in delete_result and delete_result["status"] != 200, "User should not be able to delete keys for team"
        
        # User tries regenerating a key with team_id
        regenerate_result = await regenerate_key(
            session=session,
            key=user_key,
            key_id=key_id,
            team_id=team_id
        )
        assert "status" in regenerate_result and regenerate_result["status"] != 200, "User should not be able to regenerate keys for team"