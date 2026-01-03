import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelGroup } from "../llm_calls/fetch_models";
import { determineEndpointType } from "./EndpointUtils";
import { EndpointType } from "./mode_endpoint_mapping";

// Mock the getEndpointType function
vi.mock("./mode_endpoint_mapping", () => ({
  EndpointType: {
    IMAGE: "image",
    VIDEO: "video",
    CHAT: "chat",
    RESPONSES: "responses",
    IMAGE_EDITS: "image_edits",
    ANTHROPIC_MESSAGES: "anthropic_messages",
    EMBEDDINGS: "embeddings",
    SPEECH: "speech",
    TRANSCRIPTION: "transcription",
    A2A_AGENTS: "a2a_agents",
  },
  getEndpointType: vi.fn(),
  ModelMode: {
    AUDIO_SPEECH: "audio_speech",
    AUDIO_TRANSCRIPTION: "audio_transcription",
    IMAGE_GENERATION: "image_generation",
    VIDEO_GENERATION: "video_generation",
    CHAT: "chat",
    RESPONSES: "responses",
    IMAGE_EDITS: "image_edits",
    ANTHROPIC_MESSAGES: "anthropic_messages",
    EMBEDDING: "embedding",
  },
}));

// Import the mocked function
import { getEndpointType } from "./mode_endpoint_mapping";

describe("determineEndpointType", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return the correct endpoint type when model is found and has a valid mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
      {
        model_group: "dall-e-3",
        mode: "image_generation",
      },
    ];

    // Mock getEndpointType to return IMAGE for image_generation mode
    vi.mocked(getEndpointType).mockReturnValue(EndpointType.IMAGE);

    const result = determineEndpointType("dall-e-3", mockModelInfo);

    expect(getEndpointType).toHaveBeenCalledWith("image_generation");
    expect(result).toBe(EndpointType.IMAGE);
  });

  it("should return CHAT endpoint type when model is found but has no mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        // No mode property
      },
    ];

    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should return CHAT endpoint type when model is not found in modelInfo", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
    ];

    const result = determineEndpointType("non-existent-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should return CHAT endpoint type when modelInfo array is empty", () => {
    const mockModelInfo: ModelGroup[] = [];

    const result = determineEndpointType("any-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle different mode types correctly", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "tts-model",
        mode: "audio_speech",
      },
      {
        model_group: "whisper-model",
        mode: "audio_transcription",
      },
      {
        model_group: "embedding-model",
        mode: "embedding",
      },
      {
        model_group: "video-model",
        mode: "video_generation",
      },
    ];

    // Test speech mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.SPEECH);
    const speechResult = determineEndpointType("tts-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("audio_speech");
    expect(speechResult).toBe(EndpointType.SPEECH);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test transcription mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.TRANSCRIPTION);
    const transcriptionResult = determineEndpointType("whisper-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("audio_transcription");
    expect(transcriptionResult).toBe(EndpointType.TRANSCRIPTION);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test embedding mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.EMBEDDINGS);
    const embeddingResult = determineEndpointType("embedding-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("embedding");
    expect(embeddingResult).toBe(EndpointType.EMBEDDINGS);

    // Reset mock for next test
    vi.clearAllMocks();

    // Test video mode
    vi.mocked(getEndpointType).mockReturnValueOnce(EndpointType.VIDEO);
    const videoResult = determineEndpointType("video-model", mockModelInfo);
    expect(getEndpointType).toHaveBeenCalledWith("video_generation");
    expect(videoResult).toBe(EndpointType.VIDEO);
  });

  it("should prioritize the first matching model when there are duplicates", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "gpt-3.5-turbo",
        mode: "chat",
      },
      {
        model_group: "gpt-3.5-turbo",
        mode: "image_generation", // Different mode for same model name
      },
    ];

    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);

    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).toHaveBeenCalledWith("chat");
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle models with undefined mode property explicitly set", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "test-model",
        mode: undefined,
      },
    ];

    const result = determineEndpointType("test-model", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle models with empty string mode", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "test-model",
        mode: "",
      },
    ];

    const result = determineEndpointType("test-model", mockModelInfo);

    // Empty string is falsy, so getEndpointType should not be called
    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });

  it("should handle case-sensitive model group matching", () => {
    const mockModelInfo: ModelGroup[] = [
      {
        model_group: "GPT-3.5-TURBO",
        mode: "chat",
      },
    ];

    vi.mocked(getEndpointType).mockReturnValue(EndpointType.CHAT);

    // Test with different case - should not match
    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });
});
