import { test, expect } from "@playwright/test";
import { loginToUI } from "../utils/login";

test.describe("Invite User, Set Password, and Login", () => {
  let testEmail: string;
  const testPassword = "Password123!"; // Define a password
  const teamName1 = `team-invite-test-1-${Date.now()}`;
  const teamName2 = `team-invite-test-2-${Date.now()}`;
  const keyName1 = `key-${teamName1}`;
  const keyName2 = `key-${teamName2}`;

  test.beforeEach(async ({ page }) => {
    console.log("Starting beforeEach hook: Logging in as admin...");
    await loginToUI(page); // Login as admin first
    console.log("Logged in as admin. Navigating to Teams page...");
    await page.goto("http://localhost:4000/ui?page=teams");

    // --- Create Team 1 ---
    console.log("Waiting for '+ Create New Team' button to be visible...");
    await page.getByRole("button", { name: "+ Create New Team" }).waitFor({ state: "visible", timeout: 10000 }); // Added wait
    console.log("Creating Team 1...");
    await page.getByRole("button", { name: "+ Create New Team" }).click();
    await page
      .getByLabel("Team Name")
      .waitFor({ state: "visible", timeout: 5000 }); // Wait for label
    console.log("Filling Team 1 name...");
    await page.getByLabel("Team Name").click();
    await page.getByLabel("Team Name").fill(teamName1);
    console.log("Submitting Create Team 1 form...");
    await page.getByRole("button", { name: "Create Team" }).click();
    // Wait for the modal to close or for a success message if applicable
    console.log("Waiting for Create Team 1 modal to close...");
    await expect(
      page.locator(".ant-modal-wrap").filter({ hasText: "Create New Team" })
    ).not.toBeVisible({ timeout: 10000 });
    console.log(`Created Team 1: ${teamName1}`);

    // --- Create Team 2 ---
    console.log("Creating Team 2...");
    console.log("Waiting for '+ Create New Team' button to be visible...");
    await page.getByRole("button", { name: "+ Create New Team" }).waitFor({ state: "visible", timeout: 10000 }); // Added wait
    await page.getByRole("button", { name: "+ Create New Team" }).click();
    await page
      .getByLabel("Team Name")
      .waitFor({ state: "visible", timeout: 5000 }); // Wait for label
    console.log("Filling Team 2 name...");
    await page.getByLabel("Team Name").click();
    await page.getByLabel("Team Name").fill(teamName2);
    console.log("Submitting Create Team 2 form...");
    await page.getByRole("button", { name: "Create Team" }).click();
    // Wait for the modal to close or for a success message if applicable
    console.log("Waiting for Create Team 2 modal to close...");
    await expect(
      page.locator(".ant-modal-wrap").filter({ hasText: "Create New Team" })
    ).not.toBeVisible({ timeout: 10000 });
    console.log(`Created Team 2: ${teamName2}`);

    // // Verify both teams are listed
    // await page.goto("http://localhost:4000/ui?page=teams"); // Refresh or ensure on teams page
    // await page.waitForTimeout(3000);
    console.log("Verifying Team 1 is visible...");
    await expect(page.getByText(teamName1)).toBeVisible({ timeout: 10000 });
    console.log("Verifying Team 2 is visible...");
    await expect(page.getByText(teamName2)).toBeVisible({ timeout: 10000 });

    // --- Navigate to Keys Page ---
    console.log("Navigating to API Keys page...");
    await page.goto("http://localhost:4000/ui?page=api-keys");
    console.log("Waiting for API Keys page to load...");
    await page.waitForTimeout(3000);
    await expect(
      page.getByRole("button", { name: "+ Create New Key" })
    ).toBeVisible(); // Wait for page load

    // --- Create Key for Team 1 ---
    console.log("Creating Key 1 for Team 1...");
    await page.getByRole("button", { name: "+ Create New Key" }).click();
    const createKeyModal1 = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Key Ownership" });
    console.log("Waiting for Create Key 1 modal...");
    await expect(createKeyModal1).toBeVisible();

    // Select Team 1
    console.log("Selecting Team 1 in dropdown...");
    await createKeyModal1
      .locator(".ant-select-selector >> input")
      .first()
      .click(); // Click to open team dropdown
    await createKeyModal1
      .locator(".ant-select-selector >> input")
      .first()
      .fill(teamName1);

    await page
      .locator(".ant-select-item-option")
      .filter({ hasText: teamName1 })
      .first()
      .click(); // Click specific team name

    // Enter Key Name 1
    console.log("Entering Key 1 alias...");
    await page.fill('input[id="key_alias"]', keyName1);

    // Click on models dropdown
    console.log("Selecting models for Key 1...");
    await page.locator("input#models").click();
    await page.waitForSelector(
      '.ant-select-item-option[title="All Team Models"]'
    );
    await page
      .locator('.ant-select-item-option[title="All Team Models"]')
      .click();

    // Click Create Key
    console.log("Submitting Create Key 1...");
    await createKeyModal1.getByRole("button", { name: "Create Key" }).click();

    // Close the Key Generated modal (which appears after successful creation)
    console.log("Waiting for Key 1 Generated modal...");
    const keyGeneratedModal1 = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Save your Key" });
    await expect(keyGeneratedModal1).toBeVisible({ timeout: 10000 });
    console.log("Closing Key 1 Generated modal...");
    await keyGeneratedModal1.locator('button[aria-label="Close"]').click();
    await expect(keyGeneratedModal1).not.toBeVisible(); // Wait for close
    console.log(`Created Key 1: ${keyName1} for Team: ${teamName1}`);

    // --- Create Key for Team 2 ---
    console.log("Creating Key 2 for Team 2...");
    await page.getByRole("button", { name: "+ Create New Key" }).click();
    const createKeyModal2 = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Key Ownership" });
    console.log("Waiting for Create Key 2 modal...");
    await expect(createKeyModal2).toBeVisible();

    // Select Team 2
    console.log("Selecting Team 2 in dropdown...");
    await createKeyModal2
      .locator(".ant-select-selector >> input")
      .first()
      .click(); // Click to open team dropdown
    await page
      .locator(".ant-select-item-option")
      .filter({ hasText: teamName2 })
      .click(); // Click specific team name

    // Enter Key Name 2
    console.log("Entering Key 2 alias...");
    await page.fill('input[id="key_alias"]', keyName2);

    // Click on models dropdown
    console.log("Selecting models for Key 2...");
    await page.locator("input#models").click();
    await page.waitForSelector(
      '.ant-select-item-option[title="All Team Models"]'
    );
    await page
      .locator('.ant-select-item-option[title="All Team Models"]')
      .click();

    // Click Create Key
    console.log("Submitting Create Key 2...");
    await createKeyModal2.getByRole("button", { name: "Create Key" }).click();

    // Close the Key Generated modal
    console.log("Waiting for Key 2 Generated modal...");
    const keyGeneratedModal2 = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Save your Key" });
    await expect(keyGeneratedModal2).toBeVisible({ timeout: 10000 });
    console.log("Closing Key 2 Generated modal...");
    await keyGeneratedModal2.locator('button[aria-label="Close"]').click();
    await expect(keyGeneratedModal2).not.toBeVisible(); // Wait for close
    console.log(`Created Key 2: ${keyName2} for Team: ${teamName2}`);
    console.log("Finished beforeEach hook.");
  });

  test("Invite user, set password via link, and login", async ({ page }) => {
    console.log("Starting test: Invite user, set password via link, and login");
    // Navigate to Users page
    console.log("Navigating to Users page...");
    await page.goto("http://localhost:4000/ui?page=users");

    // Go to Internal User tab
    console.log("Switching to Internal User tab...");
    const internalUserTab = page.locator("span.ant-menu-title-content", {
      hasText: "Internal User",
    });
    await internalUserTab.waitFor({ state: "visible", timeout: 10000 });
    await internalUserTab.click();

    // --- Invite User Flow ---
    console.log("Clicking Invite User button...");
    await page.getByRole("button", { name: "+ Invite User" }).click();

    // Wait for the invite user modal to be visible
    console.log("Waiting for Invite User modal...");
    const inviteModal = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Invite User" });
    await expect(inviteModal).toBeVisible();

    testEmail = `test-${Date.now()}@litellm.ai`; // Use a unique email
    console.log(`Filling email: ${testEmail}...`);
    // Assuming the email input is the first one with 'base-input' test id inside the modal
    await inviteModal.getByTestId("base-input").first().fill(testEmail);

    // Select Global Admin Role (or another appropriate role)
    console.log("Selecting Global Proxy Role...");
    const globalRoleLabel = inviteModal.getByLabel("Global Proxy Role");
    await globalRoleLabel.click();
    // Wait for the dropdown option to be visible before clicking
    console.log("Waiting for Admin role option...");
    const adminRoleOption = page.getByTitle("Admin (All Permissions)", {
      exact: true,
    });
    await adminRoleOption.waitFor({ state: "visible", timeout: 5000 });
    await adminRoleOption.click();

    // Select Team - Add explicit wait before clicking
    console.log("Selecting Team ID...");
    const teamIdLabel = inviteModal.getByLabel("Team ID");
    // Wait for the label associated with the Team ID select to be visible
    await teamIdLabel.waitFor({ state: "visible", timeout: 10000 }); // Increased timeout for safety
    await teamIdLabel.click();

    // Wait for the team name option to be visible in the dropdown
    console.log(`Waiting for team name option: ${teamName1}...`);
    const teamNameOption = page.getByText(teamName1, { exact: true });
    await teamNameOption.waitFor({ state: "visible", timeout: 5000 });
    await teamNameOption.click();

    // Create User
    console.log("Clicking Create User button...");
    await inviteModal.getByRole("button", { name: "Create User" }).click();

    // --- Capture Invitation Link ---
    console.log("Waiting for Invitation Link modal...");
    const invitationModal = page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Invitation Link" });
    await expect(invitationModal).toBeVisible({ timeout: 15000 }); // Wait longer for modal

    // Locate the text element containing the URL more reliably
    console.log("Capturing invitation URL...");
    const invitationUrl = await page
      .locator("div.flex.justify-between.pt-5.pb-2") // find the correct div
      .filter({ hasText: "Invitation Link" }) // find the div that has text "Invitation Link"
      .locator("p") // find all <p> inside that div
      .nth(1) // pick the second <p> (index 1)
      .innerText();
    console.log(`Invitation URL captured: ${invitationUrl}`);

    // Close Invitation Link Modal
    console.log("Closing Invitation Link modal...");
    await page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Invitation Link" })
      .locator('button[aria-label="Close"]')
      .click();

    // Close Invite User Modal
    console.log("Closing Invite User modal...");
    await page
      .locator(".ant-modal-wrap")
      .filter({ hasText: "Invite User" })
      .locator('button[aria-label="Close"]')
      .click();

    // Open invite link as new page (simulate invited user)
    console.log("Opening invitation link in a new context...");
    const context = await page.context()?.browser()?.newContext();
    const invitedUserPage = await context?.newPage();
    if (!invitedUserPage) {
      throw new Error("invitedUserPage is undefined");
    }
    await invitedUserPage?.goto(invitationUrl || "");
    console.log("Invitation link opened.");

    //Insert new password
    console.log("Filling password on signup page...");
    await invitedUserPage?.fill("input#password", testPassword);

    //Click on submit
    console.log("Clicking Sign Up button...");
    await invitedUserPage?.getByRole("button", { name: "Sign Up" }).click();
    console.log("Sign up submitted.");

    // // --- Verify Keys Created ---
    // console.log("Waiting for table selector after sign up...");
    // await invitedUserPage?.waitForSelector("table");
    // console.log("Table selector found.");

    // // Verify keyName1 (associated with user's team) IS visible in the table
    // console.log("Locating key table...");
    // const keyTable = invitedUserPage.locator('table'); // Locate the table element
    // await expect(keyTable).toBeVisible({ timeout: 10000 }); // Ensure table exists
    // console.log(`Verifying key ${keyName1} is visible...`);
    // // Use getByText within the table scope to find the key name
    // await expect(keyTable.getByText(keyName1, { exact: true })).toBeVisible({ timeout: 10000 });
    // console.log(`Verified key ${keyName1} is visible for user ${testEmail}`);

    // // Verify keyName2 (associated with the *other* team) IS NOT visible
    // console.log(`Verifying key ${keyName2} is NOT visible...`);
    // await expect(keyTable.getByText(keyName2, { exact: true })).not.toBeVisible();
    // console.log(`Verified key ${keyName2} is NOT visible for user ${testEmail}`);
    console.log("Test finished.");
  });
});
