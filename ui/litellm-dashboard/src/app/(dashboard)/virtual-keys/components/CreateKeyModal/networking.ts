import {
  fetchMCPAccessGroups,
  getGuardrailsList,
  getPromptsList,
  modelAvailableCall,
  User,
  userFilterUICall,
} from "@/components/networking";
import { ModelAvailableResponse, UserOption } from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/types";

export const fetchGuardrails = async (accessToken: string) => {
  try {
    const response = await getGuardrailsList(accessToken);
    return response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
  } catch (error) {
    console.error("Failed to fetch guardrails:", error);
    return [];
  }
};

export const fetchPrompts = async (accessToken: string) => {
  try {
    const response = await getPromptsList(accessToken);
    return response.prompts.map((prompt) => prompt.prompt_id);
  } catch (error) {
    console.error("Failed to fetch prompts:", error);
    return [];
  }
};

export const searchUserOptionsByEmail = async (accessToken: string, emailQuery: string): Promise<UserOption[]> => {
  const params = new URLSearchParams();
  params.append("user_email", emailQuery);

  const response = await userFilterUICall(accessToken, params);
  const users: User[] = response;

  return users.map((user) => ({
    label: `${user.user_email} (${user.user_id})`,
    value: user.user_id,
    user,
  }));
};

export const getUserModelNames = async (userID: string, userRole: string, accessToken: string): Promise<string[]> => {
  const res: ModelAvailableResponse = await modelAvailableCall(accessToken, userID, userRole);
  return (res?.data ?? []).map((m) => m.id);
};

export const fetchTeamModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  teamID: string | null,
): Promise<string[]> => {
  try {
    if (userID === null || userRole === null) {
      return [];
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole, true, teamID, true);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      console.log("available_model_names:", available_model_names);
      return available_model_names;
    }
    return [];
  } catch (error) {
    console.error("Error fetching user models:", error);
    return [];
  }
};

export const getMCPAccessGroups = async (accessToken: string): Promise<string[]> => {
  try {
    if (accessToken == null) {
      return [];
    }
    return await fetchMCPAccessGroups(accessToken);
  } catch (error) {
    console.error("Failed to fetch MCP access groups:", error);
    return [];
  }
};
