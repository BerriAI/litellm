import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test(`10109_Create_New_User_With_Team_as_User_Member`, async ({
  loginPage,
  dashboardLinks,
  teamsPage,
  internalUsersPage,
  page,
}) => {
  let username = "admin";
  let password = "sk-1234";

  if (loginDetailsSet()) {
    console.log("Login details exist in .env file.");
    username = process.env.UI_USERNAME as string;
    password = process.env.UI_PASSWORD as string;
  }

  console.log("1. Navigating to 'Login' page and logging in");
  await loginPage.goto();

  await loginPage.login(username, password);
  await expect(page.getByRole("button", { name: "User" })).toBeVisible();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/01_dashboard.png`,
  });
  await page.pause();

  /////////
  await page.goto("http://localhost:3000/ui");
  /////////

  console.log("2. Navigate to 'Teams' page.");
  await dashboardLinks.getTeamsPageLink().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/02_teams_page.png`,
  });

  console.log("3. Click '+ Create New Team'.");
  await teamsPage.getCreateNewTeamButton().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/03_click_create_team.png`,
  });

  // 4. Enter team name in 'Team Name' field.
  console.log("4. Enter team name in 'Team Name' field.");
  await teamsPage.getTeamNameField().click();
  await teamsPage.getTeamNameField().fill("team_1");
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/04_enter_team_name.png`,
  });

  // 5. Click 'Create Team'.
  console.log("5. Clicking 'Create Team'.");
  await teamsPage.getCreateTeamButton().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/05_click_create_team.png`,
  });

  // 6. Navigate to 'Internal Users' page.
  console.log("6. Navigating to 'Internal Users' page.");
  await dashboardLinks.getInternalUsersPageLink().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/06_internal_users_page.png`,
  });

  // 7. Click the '+ Invite User' button.
  console.log("7. Clicking '+ Invite User'.");
  await internalUsersPage.getInviteUsersButton().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/07_invite_user.png`,
  });

  // 8. Enter email in 'User Email' field.
  console.log("8. Entering email in 'User Email' field.");
  await internalUsersPage.getInviteUserFormUserEmailField().click();
  await internalUsersPage
    .getInviteUserFormUserEmailField()
    .fill("10109_test2@berri.ai");
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/08_user_email.png`,
  });

  // 9. Enter team id in 'Team ID' field.
  console.log("9. Entering team id in 'Team ID' field.");
  await internalUsersPage.getInviteUserFormTeamIdField().click();
  await page.getByText("team_1").click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/09_team_id.png`,
  });

  // 10. Select Team Role in 'Team Role' field.
  console.log("10. Select Team Role in 'Team Role' field.");
  await internalUsersPage.getInviteUserFormTeamRoleField().click();
  await internalUsersPage
    .getInviteUserFormTeamRoleSelection("user- User role. Can view")
    .click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/10_team_role.png`,
  });

  // 11. Click 'Create User'.
  console.log("11. Clicking 'Create User'.");
  await internalUsersPage.getInviteUserFormCreateUserButton().click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/11_create_user.png`,
  });

  // 12. Copy invitation link.
  console.log("12. Copying invitation link.");
  let invitationLink = await page
    .getByText("http://localhost:3000/ui?invitation_id")
    .innerText();
  console.log("Invitation Link Text: " + invitationLink);
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/12_copy_invitation_link.png`,
  });

  // 13. Open new tab.
  // 14. Navigate to inviation link.
  // 15. Enter password.
  // 16. Go to 'Team' page.
  // 17. Verify user is a member of team created.
});

/*
 await page.goto('http://localhost:3000/ui');
  await page.goto('http://localhost:4000/sso/key/generate');
  await page.getByText('Teams').click();
  await page.getByRole('button', { name: '+ Create New Team' }).click();
  await page.getByLabel('Team Name').click();
  await page.getByLabel('Team Name').fill('team_1');
  await page.getByRole('button', { name: 'Create Team' }).click();
  await page.getByText('Internal Users').click();
  await page.getByRole('button', { name: '+ Invite User' }).click();
  await page.getByTestId('base-input').click();
  await page.getByTestId('base-input').fill('test1@berri.ai');
  await page.getByLabel('Team ID').click();
  await page.getByText('team_1').click();
  await page.getByLabel('Team Role').click();
  await page.getByText('user- User role. Can view').click();
  await page.getByRole('button', { name: 'Create User' }).click();
  await page.getByText('Invitation Linkhttp://').click();
  await page.getByRole('button', { name: 'Copy invitation link' }).click();
  await page1.goto('http://localhost:3000/ui?invitation_id=2689689d-3e44-4bc0-a3c4-569777db6b8b');
  await page1.getByLabel('Password', { exact: true }).click();
  await page1.getByLabel('Password', { exact: true }).fill('password');
  await page1.getByRole('button', { name: 'Sign Up' }).click();
  await page1.getByText('Teams').click();
  await page1.getByRole('cell', { name: 'team_1' }).click();
  await page1.getByRole('button', { name: 'User' }).click();
  await page1.getByText('Logout').click();
  await page1.getByRole('heading', { name: 'Login' }).click();
*/
