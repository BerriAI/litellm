import {
  createKey,
  updateKey,
  deleteKey,
  createTeam,
  updateTeam,
  deleteTeam
} from '../api/dataManagement'; // Adjust path as needed

// Replace with your actual access token and user ID
const accessToken = "YOUR_ACCESS_TOKEN";
// userID is typically required for creating keys, associate them with a user.
const userID = "YOUR_USER_ID";

// ----- Key Management Examples -----

interface KeyData {
  key_alias?: string;
  team_id?: string;
  models?: string[]; // e.g., ["gpt-3.5-turbo", "claude-2"]
  spend?: number; // initial spend (usually 0)
  max_budget?: number; // e.g., 100.0
  budget_duration?: string; // e.g., "30d", "1m"
  metadata?: Record<string, any>; // e.g., { project: "alpha", description: "Test key" }
  key_name?: string; // often same as key_alias, used in some contexts
  token?: string; // The key/token string itself, needed for update/delete operations
  // Add any other fields relevant for your key creation/update API
  // such as tpm_limit, rpm_limit, user_id (if not passed separately), etc.
}

const keysToCreate: KeyData[] = [
  { key_alias: "bulk-key-alpha", models: ["gpt-3.5-turbo"], max_budget: 10, budget_duration: "30d", metadata: { project: "AlphaProject" } },
  { key_alias: "bulk-key-beta", team_id: "TEAM_ID_FOR_KEY_BETA", models: ["claude-2", "gpt-4"], metadata: { project: "BetaProject", priority: "high" } },
  // Add more key data objects here
];

// For updates, you need the 'token' (the actual key string) of the key you want to modify.
const keysToUpdate: KeyData[] = [
  { token: "EXISTING_KEY_TOKEN_1", key_alias: "bulk-key-alpha-updated", max_budget: 25, metadata: { project: "AlphaProject", status: "active" } },
  { token: "EXISTING_KEY_TOKEN_2", models: ["gpt-4-turbo"], metadata: { project: "BetaProject", priority: "critical" } },
  // Add more key update data objects here
];

// For deletion, you need the 'token' (the actual key string).
const keysToDelete: string[] = [
  "KEY_TOKEN_TO_DELETE_1",
  "KEY_TOKEN_TO_DELETE_2",
  // Add more key tokens to delete here
];

const handleCreateMultipleKeys = async () => {
  console.log("Attempting to create multiple keys...");
  for (const keyData of keysToCreate) {
      try {
          // createKey from dataManagement re-exports keyCreateCall which expects (accessToken, userID, formValues)
          const result = await createKey(accessToken, userID, keyData);
          console.log(`Key created: ${result.key_alias || (result as any).name || 'N/A'}`, result);
      } catch (error) {
          console.error(`Failed to create key ${keyData.key_alias || 'N/A'}:`, error);
      }
  }
};

const handleUpdateMultipleKeys = async () => {
  console.log("Attempting to update multiple keys...");
  for (const keyData of keysToUpdate) {
      if (!keyData.token) {
          console.error("Key token (actual key string) is required for update:", keyData);
          continue;
      }
      try {
          // updateKey from dataManagement re-exports keyUpdateCall which expects (accessToken, formValues)
          // formValues should contain the key 'token' and other fields to update.
          const result = await updateKey(accessToken, keyData);
          console.log(`Key updated: ${keyData.token}`, result);
      } catch (error) {
          console.error(`Failed to update key ${keyData.token}:`, error);
      }
  }
};

const handleDeleteMultipleKeys = async () => {
  console.log("Attempting to delete multiple keys...");
  for (const keyToken of keysToDelete) {
      try {
          // deleteKey from dataManagement re-exports keyDeleteCall which expects (accessToken, user_key)
          await deleteKey(accessToken, keyToken);
          console.log(`Key deleted: ${keyToken}`);
      } catch (error) {
          console.error(`Failed to delete key ${keyToken}:`, error);
      }
  }
};

// ----- Team Management Examples -----

interface TeamData {
  team_id?: string;       // Required for update/delete. Generated on creation.
  team_alias?: string;    // e.g., "Development Team"
  models?: string[];      // e.g., ["gpt-4", "claude-3"]
  tpm_limit?: number;
  rpm_limit?: number;
  max_budget?: number;
  budget_duration?: string; // e.g., "30d", "1m"
  admins?: string[];      // Array of user_ids for team admins
  members?: string[];     // Array of user_ids for team members
  metadata?: Record<string, any>; // e.g., { department: "engineering", project_code: "X7" }
  organization_id?: string; // If teams are part of organizations
  // Add any other fields relevant for your team creation/update API
}

const teamsToCreate: TeamData[] = [
  { team_alias: "Bulk Team Engineering", models: ["gpt-4-turbo"], admins: [userID], metadata: { department: "R&D" } },
  { team_alias: "Bulk Team Marketing", max_budget: 500, budget_duration: "60d", metadata: { campaign: "Q3-Launch" } },
  // Add more team data objects here
];

// For updates, you need the 'team_id' of the team you want to modify.
const teamsToUpdate: TeamData[] = [
  { team_id: "EXISTING_TEAM_ID_1", team_alias: "Bulk Team Engineering (Updated)", tpm_limit: 10000 },
  { team_id: "EXISTING_TEAM_ID_2", metadata: { campaign: "Q3-Launch-Revised", budget_owner: "some_user_id" } },
  // Add more team update data objects here
];

// For deletion, you need the 'team_id'.
const teamsToDelete: string[] = [
  "TEAM_ID_TO_DELETE_1",
  "TEAM_ID_TO_DELETE_2",
  // Add more team IDs to delete here
];

const handleCreateMultipleTeams = async () => {
  console.log("Attempting to create multiple teams...");
  for (const teamData of teamsToCreate) {
      try {
          // createTeam from dataManagement re-exports teamCreateCall which expects (accessToken, formValues)
          const result = await createTeam(accessToken, teamData);
          console.log(`Team created: ${result.team_alias || (result as any).name || 'N/A'} with ID: ${result.team_id}`, result);
      } catch (error) {
          console.error(`Failed to create team ${teamData.team_alias || 'N/A'}:`, error);
      }
  }
};

const handleUpdateMultipleTeams = async () => {
  console.log("Attempting to update multiple teams...");
  for (const teamData of teamsToUpdate) {
      if (!teamData.team_id) {
          console.error("Team ID is required for update:", teamData);
          continue;
      }
      try {
          // updateTeam from dataManagement re-exports teamUpdateCall which expects (accessToken, formValues)
          // formValues should include the team_id and other fields to update.
          const result = await updateTeam(accessToken, teamData);
          console.log(`Team updated: ${teamData.team_id}`, result);
      } catch (error) {
          console.error(`Failed to update team ${teamData.team_id}:`, error);
      }
  }
};

const handleDeleteMultipleTeams = async () => {
  console.log("Attempting to delete multiple teams...");
  for (const teamId of teamsToDelete) {
      try {
          // deleteTeam from dataManagement re-exports teamDeleteCall which expects (accessToken, teamID)
          await deleteTeam(accessToken, teamId);
          console.log(`Team deleted: ${teamId}`);
      } catch (error) {
          console.error(`Failed to delete team ${teamId}:`, error);
      }
  }
};

// ----- Execution -----
// This function can be called to run all the batch operations.
// IMPORTANT:
// 1. Fill in "YOUR_ACCESS_TOKEN" and "YOUR_USER_ID".
// 2. Fill in actual "TEAM_ID_FOR_KEY_BETA", "EXISTING_KEY_TOKEN_1", "EXISTING_TEAM_ID_1", etc.,
//    with REAL IDs/tokens from your system if you are updating or deleting existing entities.
// 3. For creation, ensure the data (like team_id for keys) is valid if referenced.

const runAllTestDataOperations = async () => {
  if (accessToken === "YOUR_ACCESS_TOKEN" || userID === "YOUR_USER_ID") {
      console.warn("Please replace placeholder ACCESS_TOKEN and USER_ID in test.tsx before running operations.");
      return;
  }

  console.log("--- Starting Key Creation ---");
  await handleCreateMultipleKeys();

  // ---- IMPORTANT: ----
  // You'll likely want to fetch the newly created key tokens and team IDs
  // if you intend to immediately update/delete them in the same script run.
  // For simplicity, the update/delete arrays below use placeholders.
  // You would typically:
  // 1. Create items.
  // 2. Get their IDs/tokens from the creation results.
  // 3. Use those actual IDs/tokens for update/delete operations.

  console.log("\\n--- Starting Key Updates (using placeholder tokens) ---");
  // Make sure 'keysToUpdate' contains valid, existing key tokens.
  // await handleUpdateMultipleKeys();

  console.log("\\n--- Starting Key Deletion (using placeholder tokens) ---");
  // Make sure 'keysToDelete' contains valid, existing key tokens.
  // await handleDeleteMultipleKeys();

  console.log("\\n--- Starting Team Creation ---");
  await handleCreateMultipleTeams();

  console.log("\\n--- Starting Team Updates (using placeholder IDs) ---");
  // Make sure 'teamsToUpdate' contains valid, existing team IDs.
  // await handleUpdateMultipleTeams();

  console.log("\\n--- Starting Team Deletion (using placeholder IDs) ---");
  // Make sure 'teamsToDelete' contains valid, existing team IDs.
  // await handleDeleteMultipleTeams();

  console.log("\\n--- Test data operations complete ---");
  console.log("Review console logs for details. Update/Delete operations might be commented out or use placeholders.");
  console.log("Ensure you replace placeholder IDs/tokens with actual data from your system for updates/deletions.");
};

// To run these operations:
// 1. Ensure you have a valid `accessToken` and `userID`.
// 2. If updating/deleting, ensure the placeholder IDs/tokens in `keysToUpdate`,
//    `keysToDelete`, `teamsToUpdate`, `teamsToDelete` are replaced with actual existing IDs/tokens.
// 3. Uncomment the call to `runAllTestDataOperations()` below or call it from elsewhere.

// runAllTestDataOperations();

// You can also export these functions if you want to call them individually, perhaps from a UI for testing.
export {
  handleCreateMultipleKeys,
  handleUpdateMultipleKeys,
  handleDeleteMultipleKeys,
  handleCreateMultipleTeams,
  handleUpdateMultipleTeams,
  handleDeleteMultipleTeams,
  runAllTestDataOperations
};