
import { modelAvailableCall } from "../networking";


export const fetchAvailableModelsForTeamOrKey = async (
  userID: string,
  userRole: string,
  accessToken: string,
): Promise<string[] | undefined> => {
      try {
        if (userID === null || userRole === null) {
          return;
        }
  
        if (accessToken !== null) {
          const model_available = await modelAvailableCall(
            accessToken,
            userID,
            userRole,
            true
          );
          
          let available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
  
          // Group and sort models
          const providerModels: string[] = [];
          const specificModels: string[] = [];

          available_model_names.forEach((model: string) => {
            if (model.endsWith('/*')) {
              providerModels.push(model);
            } else {
              specificModels.push(model);
            }
          });

          // Combine arrays with provider models first
          return [...providerModels, ...specificModels];
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

export const getModelDisplayName = (model: string) => {
  console.log("getModelDisplayName", model);
  if (model.endsWith('/*')) {
    const provider = model.replace('/*', '');
    return `All ${provider} Models`;
  }
  return model;
};

export const unfurlWildcardModelsInList = (teamModels: string[], allModels: string[]): string[] => {
  const result: string[] = [];
  console.log("teamModels", teamModels);
  console.log("allModels", allModels);
  
  teamModels.forEach(teamModel => {
    if (teamModel.endsWith('/*')) {
      // Extract the provider prefix (e.g., 'openai' from 'openai/*')
      const provider = teamModel.replace('/*', '');
      
      // Find all models that start with this provider
      const matchingModels = allModels.filter(model => 
        model.startsWith(provider + '/')
      );
      
      result.push(...matchingModels);
    } else {
      // For non-wildcard models, add them directly
      result.push(teamModel);
    }
  });
  
  // Remove duplicates and return
  return result.filter((item, index) => result.indexOf(item) === index);
};