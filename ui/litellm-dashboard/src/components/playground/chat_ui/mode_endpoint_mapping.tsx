// litellmMapping.ts

// Define an enum for the modes as returned in model_info
export enum ModelMode {
  AUDIO_SPEECH = "audio_speech",
  AUDIO_TRANSCRIPTION = "audio_transcription",
  IMAGE_GENERATION = "image_generation",
  VIDEO_GENERATION = "video_generation",
  CHAT = "chat",
  RESPONSES = "responses",
  IMAGE_EDITS = "image_edits",
  ANTHROPIC_MESSAGES = "anthropic_messages",
  EMBEDDING = "embedding",
  // add additional modes as needed
}

// Define an enum for the endpoint types your UI calls
export enum EndpointType {
  IMAGE = "image",
  VIDEO = "video",
  CHAT = "chat",
  RESPONSES = "responses",
  IMAGE_EDITS = "image_edits",
  ANTHROPIC_MESSAGES = "anthropic_messages",
  EMBEDDINGS = "embeddings",
  SPEECH = "speech",
  TRANSCRIPTION = "transcription",
  A2A_AGENTS = "a2a_agents",
  MCP = "mcp",
  // add additional endpoint types if required
}

// Create a mapping between the model mode and the corresponding endpoint type
export const litellmModeMapping: Record<ModelMode, EndpointType> = {
  [ModelMode.IMAGE_GENERATION]: EndpointType.IMAGE,
  [ModelMode.VIDEO_GENERATION]: EndpointType.VIDEO,
  [ModelMode.CHAT]: EndpointType.CHAT,
  [ModelMode.RESPONSES]: EndpointType.RESPONSES,
  [ModelMode.IMAGE_EDITS]: EndpointType.IMAGE_EDITS,
  [ModelMode.ANTHROPIC_MESSAGES]: EndpointType.ANTHROPIC_MESSAGES,
  [ModelMode.AUDIO_SPEECH]: EndpointType.SPEECH,
  [ModelMode.AUDIO_TRANSCRIPTION]: EndpointType.TRANSCRIPTION,
  [ModelMode.EMBEDDING]: EndpointType.EMBEDDINGS,
};

export const getEndpointType = (mode: string): EndpointType => {
  // Check if the string mode exists as a key in ModelMode enum
  console.log("getEndpointType:", mode);
  if (Object.values(ModelMode).includes(mode as ModelMode)) {
    const endpointType = litellmModeMapping[mode as ModelMode];
    console.log("endpointType:", endpointType);
    return endpointType;
  }

  // else default to chat
  return EndpointType.CHAT;
};
