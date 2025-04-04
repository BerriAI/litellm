import { modelAvailableCall } from "../../networking";

interface ModelOption {
  value: string;
  label: string;
}

/**
 * Fetches available models for the user and formats them as options
 * for selection dropdowns
 */
export const fetchAvailableModels = async (
  apiKey: string | null,
  userID: string,
  userRole: string,
  teamID: string | null = null
): Promise<ModelOption[]> => {
  try {
    const fetchedAvailableModels = await modelAvailableCall(
      apiKey ?? '', // Use empty string if apiKey is null
      userID,
      userRole,
      false,
      teamID
    );

    console.log("model_info:", fetchedAvailableModels);

    if (fetchedAvailableModels?.data.length > 0) {
      // Create a Map to store unique models using the model ID as key
      const uniqueModelsMap = new Map();
      
      fetchedAvailableModels["data"].forEach((item: { id: string }) => {
        uniqueModelsMap.set(item.id, {
          value: item.id,
          label: item.id
        });
      });

      // Convert Map values back to array
      const uniqueModels = Array.from(uniqueModelsMap.values());

      // Sort models alphabetically
      uniqueModels.sort((a, b) => a.label.localeCompare(b.label));

      return uniqueModels;
    }
    
    return [];
  } catch (error) {
    console.error("Error fetching model info:", error);
    throw error;
  }
}; 