import { test, expect } from "./../fixtures/fixtures";
import { loginDetailsSet } from "./../utils/utils";

test(`10109_Create_New_User_With_Team_as_User_Member`, async ({
  loginPage,
  dashboardLinks,
  modelsPage,
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

  console.log("2. Navigate to 'Teams' page.");
  await page.getByText("Teams").click();
  await page.screenshot({
    path: `./test-results/10109_Create_New_User_With_Team_as_User_Member/02_teams_page.png`,
  });

  // 3. Click '+ Create New Team'.
  // 4. Enter team name in 'Team Name' field.
  // 5. Click 'Create Team'.
  // 6. Navigate to 'Internal Users' page.
  // 7. Click the '+ Invite User' button.
  // 8. Enter email in 'User Email' field.
  // 9. Enter team role in 'Team Role' field.
  // 10. Click 'Create User'.
  // 11. Copy invitation link.
  // 12. Open new tab.
  // 13. Navigate to inviation link.
  // 14. Enter password.
  // 15. Go to 'Team' page.
  // 16. Verify user is a member of team created.
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
