// litellmMapping.ts

// Define an enum for the modes as returned in model_info
export enum ModelMode {
    IMAGE_GENERATION = "image_generation",
    CHAT = "chat",
    // add additional modes as needed
  }
  
  // Define an enum for the endpoint types your UI calls
  export enum EndpointType {
    IMAGE = "image",
    CHAT = "chat",
    // add additional endpoint types if required
  }
  
  // Create a mapping between the model mode and the corresponding endpoint type
  export const litellmModeMapping: Record<ModelMode, EndpointType> = {
    [ModelMode.IMAGE_GENERATION]: EndpointType.IMAGE,
    [ModelMode.CHAT]: EndpointType.CHAT,
  };
  