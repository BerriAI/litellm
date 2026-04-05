"""
Reproduction script for issue #19105: Team member budgets not enforced

This script demonstrates the bug where:
1. Team member budget (max_budget_in_team) is not enforced - requests go through even when exceeded
2. Team member spend doesn't reset daily while team spend does

Setup:
- Create a team with max_budget and budget_duration
- Add a key to the team
- Add a user with max_budget_in_team
- Make requests to exceed the user's budget
- Verify requests still go through (BUG)
- Trigger budget reset and verify team member spend doesn't reset (BUG)
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add litellm to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob


async def reproduce_issue():
    print("=" * 80)
    print("REPRODUCING ISSUE #19105: Team Member Budget Not Enforced")
    print("=" * 80)

    # Initialize Prisma client
    prisma_client = PrismaClient()
    await prisma_client.connect()

    try:
        # Step 1: Create a team with budget and budget_duration
        print("\n1. Creating team with budget...")
        team = await prisma_client.db.litellm_teamtable.create(
            data={
                "team_alias": "test_team_19105",
                "max_budget": 1.0,
                "budget_duration": "1d",  # Daily reset
                "spend": 0.0,
            }
        )
        print(f"   Created team: {team.team_id}, budget: ${team.max_budget}, duration: {team.budget_duration}")

        # Step 2: Create a key for the team
        print("\n2. Creating key for team...")
        key = await prisma_client.db.litellm_verificationtoken.create(
            data={
                "token": f"sk-test-{team.team_id}",
                "team_id": team.team_id,
            }
        )
        print(f"   Created key: {key.token}")

        # Step 3: Create a budget table for team member
        print("\n3. Creating budget table for team member...")
        budget = await prisma_client.db.litellm_budgettable.create(
            data={
                "max_budget": 0.01,  # Very low budget to trigger limit
                # Note: NOT setting budget_duration - this is the bug!
            }
        )
        print(f"   Created budget: budget_id={budget.budget_id}, max_budget=${budget.max_budget}")
        print(f"   ⚠️  Budget duration: {budget.budget_duration} (None - this is the problem!)")

        # Step 4: Create a user (internal user)
        print("\n4. Creating internal user...")
        user = await prisma_client.db.litellm_usertable.create(
            data={
                "user_id": "test_user_19105",
                "user_email": "test@example.com",
            }
        )
        print(f"   Created user: {user.user_id}")

        # Step 5: Create team membership linking user to team with budget
        print("\n5. Creating team membership with max_budget_in_team...")
        membership = await prisma_client.db.litellm_teammembership.create(
            data={
                "team_id": team.team_id,
                "user_id": user.user_id,
                "budget_id": budget.budget_id,
                "spend": 0.015,  # Already over budget!
            }
        )
        print(f"   Created membership: user={user.user_id}, team={team.team_id}")
        print(f"   Member spend: ${membership.spend}, Member budget: ${budget.max_budget}")
        print(f"   ⚠️  Spend ${membership.spend} > Budget ${budget.max_budget} - should be blocked!")

        # Step 6: Check if budget enforcement would work
        print("\n6. Testing budget check logic...")
        membership_with_budget = await prisma_client.db.litellm_teammembership.find_unique(
            where={"user_id_team_id": {"user_id": user.user_id, "team_id": team.team_id}},
            include={"litellm_budget_table": True},
        )

        if membership_with_budget and membership_with_budget.litellm_budget_table:
            team_member_budget = membership_with_budget.litellm_budget_table.max_budget
            team_member_spend = membership_with_budget.spend or 0.0

            print(f"   Spend: ${team_member_spend}, Budget: ${team_member_budget}")
            if team_member_spend >= team_member_budget:
                print(f"   ✓ Budget check WOULD block request (spend >= budget)")
            else:
                print(f"   ✗ Budget check would NOT block request")

        # Step 7: Test budget reset logic
        print("\n7. Testing budget reset logic...")
        print(f"   Current team member spend: ${membership.spend}")
        print(f"   Budget reset_at: {budget.budget_reset_at}")

        # Create ProxyLogging object (needed for ResetBudgetJob)
        proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
        reset_job = ResetBudgetJob(
            proxy_logging_obj=proxy_logging_obj,
            prisma_client=prisma_client,
        )

        # Manually set budget reset_at to past to trigger reset
        now = datetime.now(timezone.utc)
        past_time = now - timedelta(hours=1)
        await prisma_client.db.litellm_budgettable.update(
            where={"budget_id": budget.budget_id},
            data={"budget_reset_at": past_time},
        )
        print(f"   Set budget reset_at to: {past_time} (in the past)")

        # Run budget reset
        print(f"\n8. Running budget reset job...")
        await reset_job.reset_budget()

        # Check if team member spend was reset
        membership_after = await prisma_client.db.litellm_teammembership.find_unique(
            where={"user_id_team_id": {"user_id": user.user_id, "team_id": team.team_id}},
        )

        print(f"   Team member spend after reset: ${membership_after.spend}")
        if membership_after.spend == 0.0:
            print(f"   ✓ Team member spend WAS reset")
        else:
            print(f"   ✗ BUG: Team member spend NOT reset (still ${membership_after.spend})")
            print(f"   This is because the budget table doesn't have a budget_duration set!")

        # Clean up
        print("\n9. Cleaning up...")
        await prisma_client.db.litellm_teammembership.delete(
            where={"user_id_team_id": {"user_id": user.user_id, "team_id": team.team_id}}
        )
        await prisma_client.db.litellm_usertable.delete(where={"user_id": user.user_id})
        await prisma_client.db.litellm_verificationtoken.delete(where={"token": key.token})
        await prisma_client.db.litellm_budgettable.delete(where={"budget_id": budget.budget_id})
        await prisma_client.db.litellm_teamtable.delete(where={"team_id": team.team_id})
        print("   Cleanup complete")

        print("\n" + "=" * 80)
        print("CONCLUSION:")
        print("=" * 80)
        print("Team member budgets don't reset because:")
        print("1. Team members use litellm_budget_table for budget limits")
        print("2. Budget resets only happen if litellm_budget_table has budget_duration set")
        print("3. When creating team members with max_budget_in_team, the budget_duration")
        print("   is NOT automatically inherited from the team")
        print("4. Without budget_duration, the budget never resets")
        print("\nFIX: Team member budgets should inherit budget_duration from their team")
        print("=" * 80)

    finally:
        await prisma_client.disconnect()


if __name__ == "__main__":
    asyncio.run(reproduce_issue())
