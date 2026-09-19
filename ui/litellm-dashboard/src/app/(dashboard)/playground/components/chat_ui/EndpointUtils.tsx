import { ModelGroup } from "@/components/llm_calls/fetch_models";
import {
  EndpointType,
  getEndpointType,
  isModeCompatibleWithEndpoint,
} from "@/components/chat_ui/mode_endpoint_mapping";

export const determineEndpointType = (selectedModel: string, modelInfo: ModelGroup[]): EndpointType => {
  const selectedModelInfo = modelInfo.find((option) => option.model_group === selectedModel);

  if (selectedModelInfo?.mode) {
    return getEndpointType(selectedModelInfo.mode);
  }

  return EndpointType.CHAT;
};

export const isModelCompatibleWithEndpoint = (model: ModelGroup, endpointType: EndpointType): boolean =>
  isModeCompatibleWithEndpoint(model.mode, endpointType);

export const filterModelsForEndpoint = (models: ModelGroup[], endpointType: EndpointType): ModelGroup[] =>
  models.filter((model) => isModelCompatibleWithEndpoint(model, endpointType));
