export const SAP_DEPLOYMENT_MODEL_PREFIX = "sap/deployment/";

export interface SapDeployment {
  model_name: string;
  deployment_url: string;
}

export interface SapModelMapping {
  public_name: string;
  litellm_model: string;
}

export interface SapDeploymentSelection {
  model: string[];
  model_mappings: SapModelMapping[];
  api_base: string;
}

export const buildSapDeploymentSelection = (deployment: SapDeployment): SapDeploymentSelection => {
  const litellmModel = `${SAP_DEPLOYMENT_MODEL_PREFIX}${deployment.model_name}`;
  return {
    model: [litellmModel],
    model_mappings: [{ public_name: litellmModel, litellm_model: litellmModel }],
    api_base: deployment.deployment_url,
  };
};
