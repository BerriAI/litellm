import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";
import { generateRandomString } from "./../utils/utils";

test(`10109_Create_New_User_With_Team_as_User_Member`, async ({
  loginPage,
  dashboardLinks,
  teamsPage,
  internalUsersPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";
  let teamID = generateRandomString(6);
  let testuser = generateRandomString(6) + "@berri.ai";
  let invitationLink = "";

  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }

  console.log("1. Navigating to 'Login' page and logging in");
  await loginPage.goto();

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/01_dashboard.png`,});

  /////////////// Testing UI Locally ///////////////
  await page.goto("http://localhost:3000/ui");
  //////////////////////////////////////////////////

  console.log("2. Navigating to 'Teams' page.");
  await dashboardLinks.getTeamsPageLink().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/02_teams_page.png`,});

  console.log("3. Clicking '+ Create New Team'.");
  await teamsPage.getCreateNewTeamButton().click();
  // await page.screenshot({ path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/03_click_create_team.png`,});

  // 4. Enter team name in 'Team Name' field.
  console.log("4. Entering team name in 'Team Name' field.");
  await teamsPage.getTeamNameField().click();
  await teamsPage.getTeamNameField().fill(teamID);
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/04_enter_team_name.png`,});

  // 5. Click 'Create Team'.
  console.log("5. Clicking 'Create Team'.");
  await teamsPage.getCreateTeamButton().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/05_click_create_team.png`,});

  // 6. Navigate to 'Internal Users' page.
  console.log("6. Navigating to 'Internal Users' page.");
  await dashboardLinks.getInternalUsersPageLink().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/06_internal_users_page.png`,});

  // 7. Click the '+ Invite User' button.
  console.log("7. Clicking '+ Invite User'.");
  await internalUsersPage.getInviteUsersButton().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/07_invite_user.png`,});

  // 8. Enter email in 'User Email' field.
  console.log("8. Entering email in 'User Email' field.");
  await internalUsersPage.getInviteUserFormUserEmailField().click();
  await internalUsersPage.getInviteUserFormUserEmailField().fill(testuser);
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/08_user_email.png`,});

  // 9. Enter team id in 'Team ID' field.
  console.log("9. Entering team id in 'Team ID' field.");
  await page.waitForTimeout(3000);
  await internalUsersPage.getInviteUserFormTeamIdField().click();
  await page.getByText(teamID).click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/09_team_id.png`,});

  // 10. Select Team Role in 'Team Role' field.
  console.log("10. Select Team Role in 'Team Role' field.");
  await page.waitForTimeout(3000);
  await internalUsersPage.getInviteUserFormTeamRoleField().click();
  await internalUsersPage
    .getInviteUserFormTeamRoleSelection("user- User role. Can view")
    .click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/10_team_role.png`,});

  // 11. Click 'Create User'.
  console.log("11. Clicking 'Create User'.");
  await page.waitForTimeout(3000);
  await internalUsersPage.getInviteUserFormCreateUserButton().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/11_create_user.png`,});

  // 12. Copy invitation link.
  console.log("12. Copying invitation link.");
  await page.waitForTimeout(3000);
  invitationLink = await page
    .getByText("http://localhost:3000/ui?invitation_id")
    .innerText();
  console.log("Invitation Link Text: " + invitationLink);
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/12_copy_invitation_link.png`,});

  // 13. Navigate to inviation link.
  await page.waitForTimeout(3000);
  console.log("13. Navigating to inviation link.");
  await page.goto(invitationLink);
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/13_navigate_to_invitation_link.png`,});

  // 14. Enter password.
  console.log("14. Entering password.");
  await page.locator('input[type="password"]').fill("password");
  await page.waitForTimeout(3000);
  await page.getByRole("button", { name: "Sign Up" }).click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/14_entering_password.png`,});

  // 15. Go to 'Team' page.
  console.log("15. Going to 'Team' page.");
  await dashboardLinks.getTeamsPageLink().click();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/15_team_page.png`,});

  // 16. Verify user is a member of team created.
  console.log("16. Verify user is a member of team created.");
  await expect(page.getByRole("cell", { name: teamID })).toBeVisible();
  // await page.screenshot({path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/16_verify_team.png`,});

  // 17. Logging Out
  console.log("17. Logging Out");
  await dashboardLinks.getUserButton().click();
  await page.getByText("Logout").click();
});
