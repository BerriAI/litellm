/**
 * Team Admin
 */
import { test, expect } from '@playwright/test';
import { loginToUI } from '../utils/login';

test.describe("team admin ui test", async () => {
  test.beforeEach(async ({ page }) => {
    await loginToUI(page);
    //Navigate to Teams page
    await page.goto("http://localhost:4000/ui?page=teams")
  })

  test("Invite a user, make them team Admin and give them no access to models", async ({ page }) => {

    //Create two teams
    //Create a team
    await page.locator('button span:has-text("Create New Team")').click();

    //Enter "team test-1" in the team name input field
    await page.fill('input[id="team_alias"]', 'team test-1');

    //Create team test-1
    await page.locator('button span:has-text("Create Team")').click();

    //Create a team
    await page.locator('button span:has-text("Create New Team")').click();

    //Enter "team test-2" in the team name input field
    await page.fill('input[id="team_alias"]', 'team test-2');

    //Create team test-2
    await page.locator('button span:has-text("Create Team")').click();

    //Go to virtual keys tab
    await page.goto("http://localhost:4000/ui?page=api-keys")

    //Create one keys for team test-1 and test-2 each 
    //Create key for team test-1
    await page.getByRole('button', {name: '+ Create New Key'}).click();

    // Click on team dropdown
    await page.locator('input.ant-select-selection-search-input').first().click();

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option:has-text("team test-1")');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option:has-text("team test-1")').click();

    const keyName1 = `key-test-1`
    await page.fill('input[id="key_alias"]', keyName1);

    // Click on models dropdown
    await page.locator('input#models').click(); 

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="All Team Models"]');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option[title="All Team Models"]').click();

    // Click on create key
    await page.getByRole('button', {name: 'Create Key'}).click();

    //Close modal
    await page.locator('button[aria-label="Close"]').click();

    await page.waitForTimeout(1000);

    //Create key for team test-2
    await page.getByRole('button', {name: '+ Create New Key'}).click();

    // Click on team dropdown
    await page.locator('input.ant-select-selection-search-input').first().click();

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option:has-text("team test-2")');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option:has-text("team test-2")').click();

    const keyName2 = `key-test-2`
    await page.fill('input[id="key_alias"]', keyName2);

    // Click on models dropdown
    await page.locator('input#models').click(); 

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-2" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="All Team Models"]');

    // Click on the "team test-2" option
    await page.locator('.ant-select-item-option[title="All Team Models"]').click();

    // Click on create key
    await page.getByRole('button', {name: 'Create Key'}).click();

    //Close modal
    await page.locator('button[aria-label="Close"]').click();

    await page.waitForTimeout(1000);

    // Go to users tab
    await page.goto("http://localhost:4000/ui?page=users")

    //Wait for user tab to view
    await expect(page.locator('h1:has-text("User")')).toBeVisible();

    // Create two users
    // First user is assigned to team test-1
    // Second user is not assigned to any team

    //First User
    //Look for Invite User
    await expect(page.locator('button:has-text("Invite User")').first()).toBeVisible();

    //Click on Invite User
    await page.locator('button:has-text("Invite User")').first().click();

    //Fill invite form
    const email1 = `testuser${Date.now()}@example.com`; // create unique email
    await page.fill('input[id="user_email"]', email1);

    //Click on Global Proxy Role dropdown
    await page.click('input#user_role');

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "Admin (All Permissions)" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="Admin (All Permissions)"]');

    // Click on the "Admin (All Permissions)" option
    await page.locator('.ant-select-item-option[title="Admin (All Permissions)"]').click();

    // Open the dropdown by clicking the input
    await page.click('input#team_id');

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="team test-1"]');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option[title="team test-1"]').click();
    
    // Click on "Create User"
    await page.click('button:has-text("Create User")');

    const invitationLink = await page
    .locator('div.flex.justify-between.pt-5.pb-2') // find the correct div
    .filter({ hasText: 'Invitation Link' }) // find the div that has text "Invitation Link"
    .locator('p') // find all <p> inside that div
    .nth(1) // pick the second <p> (index 1)
    .innerText();

    //Close modal
    await page.getByRole('button', { name: 'Close' }).first().click();

    //Second User
    //Look for Invite User
    await expect(page.locator('button:has-text("Invite User")').first()).toBeVisible();

    //Click on Invite User
    await page.locator('button:has-text("Invite User")').first().click();

    //Fill invite form
    const email2 = `testuser${Date.now()}@example.com`; // create unique email
    await page.fill('input[id="user_email"]', email2);

    // Open the dropdown by clicking the input
    await page.click('input#team_id');

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="team test-1"]');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option[title="team test-1"]').click();
    
    // Click on "Create User"
    await page.click('button:has-text("Create User")');
    
    //Close modal
    const closeButton = page.getByRole('button', { name: 'Close' }).first();
    if (await closeButton.isVisible()) {
      await closeButton.waitFor({ state: 'visible' });
      console.log("close button is visible")
      await closeButton.click();
    }
    await page.waitForTimeout(1000);

    await page.goto("http://localhost:4000/ui?page=teams")

    // Wait for teams table to load
    await page.waitForSelector("table");

    //Locate the edit button in the first row
    const teamEditButton = await page.locator(
      "table tbody tr:first-child td:nth-child(9) span:first-of-type"
    )

    //Click on edit button in the first row
    await teamEditButton.click();

    //Wait for settings tab to load
    await page.waitForSelector('button span:has-text("Settings")');

    //Check for tabs
    await expect(page.locator('button:has-text("Overview")')).toBeVisible();
    await expect(page.locator('button:has-text("Members")')).toBeVisible();
    await expect(page.locator('button:has-text("Member Permissions")')).toBeVisible();
    await expect(page.locator('div[role="tablist"] button:has-text("Settings")')).toBeVisible();

    //Click on members tab
    await page.locator('button:has-text("Members")').click();

    //Locate the edit button in the last row
    await page.locator('table tbody tr:last-child td:nth-child(4) span.tremor-Icon-root').first().click();

    // Open the dropdown by clicking the input
    // Force clicking on the input field to open the dropdown
    await page.locator('input#role').click({ force: true });

    // Wait for the dropdown to be visible
    await page.waitForSelector('.rc-virtual-list');

    // Wait for the "admin" option to be visible using the title attribute
    await page.waitForSelector('.ant-select-item-option[title="Admin"]');

    // Click on the "team test-1" option
    await page.locator('.ant-select-item-option[title="Admin"]').click();
        
    // Click on "Create User"
    await page.click('button:has-text("Save Changes")');

    // Open invite link as new page (simulate invited user)
    const context = await page.context()?.browser()?.newContext();
    const invitedUserPage = await context?.newPage();
    if (!invitedUserPage) {
      throw new Error('invitedUserPage is undefined');
    }
    await invitedUserPage?.goto(invitationLink || '');

    //Insert new password
    await invitedUserPage?.fill('input#password', 'gm');

    //Click on submit
    await invitedUserPage?.getByRole('button', { name: 'Sign Up' }).click();

    await invitedUserPage?.waitForTimeout(1000);
    
    //Wait for keys table to load
    await invitedUserPage?.waitForSelector("table");

    //Check that the two new keys in team test-1 are visible
    const checkKeyName2 = await invitedUserPage?.locator("table tbody tr:first-child td:nth-child(2) span:first-of-type").textContent();

    // Perform the comparison after awaiting textContent
    await expect(checkKeyName2).toBe(keyName2);

    const checkKeyName1 = await invitedUserPage?.locator("table tbody tr:nth-child(2) td:nth-child(2) span:first-of-type").textContent();

    // Perform the comparison after awaiting textContent
    await expect(checkKeyName1).toBe(keyName1);

    //Go to team
    await invitedUserPage?.goto("http://localhost:4000/ui?page=teams")

    // Wait for users table to load
    await invitedUserPage?.waitForSelector("table");

    const teamIdButton = invitedUserPage?.locator(
      "table tbody tr:first-child td:nth-child(2) span:first-of-type"
    )

    //Click on edit button in the first row
    await teamIdButton?.click();

    //Click on members tab
    await invitedUserPage?.locator('button:has-text("Members")').click();

    //Click add member
    await invitedUserPage?.locator('button:has-text("Add Member")').click();

    //Add test user email
    await invitedUserPage?.fill('input[id="user_email"]', email2);

    // Wait for the dropdown to be visible
    await invitedUserPage?.waitForSelector('.ant-select-dropdown');

    // Click the first option
    await invitedUserPage?.locator('.ant-select-item-option').first().click();
        
    //Locate "Add Member"
    const addMemberButton = await invitedUserPage?.locator('button:has-text("Add Member")');

    // Click on "Add Member"
    await addMemberButton?.last().click();

    //Click on members tab
    await invitedUserPage?.locator('button:has-text("Members")').click();

    //Check the new user is added
    const testEmail = await invitedUserPage?.locator('table tbody tr:last-child td:nth-child(2) p').textContent();

    await expect(testEmail).toBe(email2)

    //Locate the delete button in the last row
    await invitedUserPage?.locator('table tbody tr:last-child td:nth-child(4) span.tremor-Icon-root').last().click();

    //TODO: Add, edit and delete model in team. Able to see all team models in test key dropdown.
    // //Add model
    // //Go to models
    // await invitedUserPage.goto("http://localhost:4000/ui?page=models")

    // //Click on Add Model
    // await invitedUserPage.locator('button[role="tab"] >> text=Add Model').click();

    // //Verify we are on Add Model page
    // await expect(invitedUserPage.locator('h2:has-text("Add new model")')).toBeVisible();

    // //Select provider from dropdown
    // await invitedUserPage.fill('input[id="custom_llm_provider"]', "");

    // //Wait for the dropdown to be visible
    // await invitedUserPage.waitForSelector('.ant-select-dropdown');

    // //Click the first option to add provider
    // await invitedUserPage.locator('.ant-select-item-option').first().click();

    // //LiteLLM Model Name, select one model
    // await invitedUserPage.locator('input#model').click();

    // //Wait for the dropdown to be visible
    // await invitedUserPage.waitForSelector('.ant-select-dropdown');

    // //Click the first option to add model
    // await invitedUserPage?.locator('[title="Custom Model Name (Enter below)"]').click();

    // //Insert custom model name
    // const modelName = "custom_model_name"
    // await invitedUserPage.fill('input[id="custom_model_name"]', modelName);

    // //Fill openAPI key
    // await invitedUserPage.fill('input[id="api_key"]', "abcd1234-efgh5678-ijkl9012-mnop3456");

    // // Click on team dropdown
    // await invitedUserPage.locator('input.ant-select-selection-search-input').last().click();

    // // // Wait for the dropdown to be visible
    // // await invitedUserPage.waitForSelector('.rc-virtual-list');

    // // Wait for the "team test-1" option to be visible using the title attribute
    // await invitedUserPage.waitForSelector('.ant-select-item-option:has-text("team test-1")');

    // // Click on the "team test-1" option
    // await invitedUserPage.locator('.ant-select-item-option:has-text("team test-1")').click();

    // //Add Model
    // await invitedUserPage.locator('button:has-text("Add Model")').last().click();

    // //Click on All Models tab
    // await invitedUserPage.locator('button:has-text("All Models")').click(); 

    // //Check the new model is added
    // const newModel = await invitedUserPage?.locator('table tbody tr:last-child td:nth-child(2) p').textContent();

    // await expect(newModel).toBe(modelName)

    //Able to create team key with all team models
    //Go to virtual keys tab
    await invitedUserPage.goto("http://localhost:4000/ui?page=api-keys")

    //Create key for team test-1
    await invitedUserPage.getByRole('button', {name: '+ Create New Key'}).click();

    // Click on team dropdown
    await invitedUserPage.locator('input.ant-select-selection-search-input').first().click();

    // Wait for the dropdown to be visible
    await invitedUserPage.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await invitedUserPage.waitForSelector('.ant-select-item-option:has-text("team test-1")');

    // Click on the "team test-1" option
    await invitedUserPage.locator('.ant-select-item-option:has-text("team test-1")').click();

    const keyName3 = `key-test-3`
    await invitedUserPage.fill('input[id="key_alias"]', keyName3);

    // Click on models dropdown
    await invitedUserPage.locator('input#models').click(); 

    // Wait for the dropdown to be visible
    await invitedUserPage.waitForSelector('.rc-virtual-list');

    // Wait for the "team test-1" option to be visible using the title attribute
    await invitedUserPage.waitForSelector('.ant-select-item-option[title="All Team Models"]');

    // Click on the "team test-1" option
    await invitedUserPage.locator('.ant-select-item-option[title="All Team Models"]').click();

    // Click on create key
    await invitedUserPage.getByRole('button', {name: 'Create Key'}).click();

    //Close modal
    await invitedUserPage.locator('button[aria-label="Close"]').click();

    //Verify if the key is added
    const checkKeyName3 = await invitedUserPage.locator("table tbody tr:first-child td:nth-child(2) span:first-of-type").textContent();

    // Perform the comparison after awaiting textContent
    await expect(checkKeyName3).toBe(keyName3);

  })
})