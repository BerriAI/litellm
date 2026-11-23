const CALLBACK_CONFIG_ALIASES: Record<string, string> = { s3_v2: "s3" };
const CALLBACK_UNSUPPORTED_PARAMS: Record<string, readonly string[]> = { s3: ["s3_partition_granularity"] };

interface DynamicParamConfig {
  type?: string;
  ui_name?: string;
  required?: boolean;
  options?: string[];
}

interface CallbackConfigWithParams {
  id: string;
  dynamic_params?: Record<string, DynamicParamConfig>;
}

export const findCallbackConfig = <T extends { id: string }>(
  callbackConfigs: readonly T[],
  callbackName: string | null,
): T | undefined => {
  if (!callbackName) {
    return undefined;
  }
  const configId = CALLBACK_CONFIG_ALIASES[callbackName] ?? callbackName;
  return callbackConfigs.find((config) => config.id === configId);
};

// Shared helper function to get dynamic params for a callback
export const getDynamicParamsForCallback = (
  callbackName: string | null,
  callbackConfigs: any[],
  fallbackVariables?: Record<string, any>,
): string[] => {
  if (!callbackName) {
    return fallbackVariables ? Object.keys(fallbackVariables) : [];
  }

  const callbackConfig = findCallbackConfig(callbackConfigs, callbackName);
  if (callbackConfig?.dynamic_params) {
    const unsupportedParams = CALLBACK_UNSUPPORTED_PARAMS[callbackName] ?? [];
    return Object.keys(callbackConfig.dynamic_params).filter((param) => !unsupportedParams.includes(param));
  }

  return fallbackVariables ? Object.keys(fallbackVariables) : [];
};

// Shared helper function to build callback payload
export const buildCallbackPayload = (formValues: Record<string, any>, callbackName: string) => {
  return {
    environment_variables: Object.fromEntries(Object.entries(formValues).filter(([, value]) => value !== undefined)),
    litellm_settings: {
      success_callback: [callbackName],
    },
  };
};
