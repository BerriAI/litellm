import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { EndpointType, getEndpointType, ModelMode } from "@/components/chat_ui/mode_endpoint_mapping";

const KNOWN_MODEL_MODES = new Set<string>(Object.values(ModelMode));

export const determineEndpointType = (selectedModel: string, modelInfo: ModelGroup[]): EndpointType => {
  const selectedModelInfo = modelInfo.find((option) => option.model_group === selectedModel);

  if (selectedModelInfo?.mode) {
    return getEndpointType(selectedModelInfo.mode);
  }

  return EndpointType.CHAT;
};

export const isModelCompatibleWithEndpoint = (model: ModelGroup, endpointType: EndpointType): boolean => {
  if (!model.mode) {
    return true;
  }

  if (!KNOWN_MODEL_MODES.has(model.mode)) {
    return false;
  }

  const optionEndpoint = getEndpointType(model.mode);

  if (
    endpointType === EndpointType.RESPONSES ||
    endpointType === EndpointType.ANTHROPIC_MESSAGES ||
    endpointType === EndpointType.INTERACTIONS
  ) {
    return optionEndpoint === endpointType || optionEndpoint === EndpointType.CHAT;
  }

  if (endpointType === EndpointType.IMAGE_EDITS) {
    return optionEndpoint === endpointType || optionEndpoint === EndpointType.IMAGE;
  }

  return optionEndpoint === endpointType;
};

export const filterModelsForEndpoint = (models: ModelGroup[], endpointType: EndpointType): ModelGroup[] =>
  models.filter((model) => isModelCompatibleWithEndpoint(model, endpointType));
