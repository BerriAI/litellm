import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { determineEndpointType, filterModelsForEndpoint, isModelCompatibleWithEndpoint } from "./EndpointUtils";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";

vi.mock("@/components/chat_ui/mode_endpoint_mapping", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/chat_ui/mode_endpoint_mapping")>();
  return {
    ...actual,
    getEndpointType: vi.fn(actual.getEndpointType),
  };
});

import { getEndpointType } from "@/components/chat_ui/mode_endpoint_mapping";

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

    const result = determineEndpointType("gpt-3.5-turbo", mockModelInfo);

    expect(getEndpointType).not.toHaveBeenCalled();
    expect(result).toBe(EndpointType.CHAT);
  });
});

describe("isModelCompatibleWithEndpoint / filterModelsForEndpoint", () => {
  beforeEach(async () => {
    const actual = await vi.importActual<typeof import("@/components/chat_ui/mode_endpoint_mapping")>(
      "@/components/chat_ui/mode_endpoint_mapping",
    );
    vi.mocked(getEndpointType).mockImplementation(actual.getEndpointType);
  });

  it("keeps models with no mode for every endpoint", () => {
    const model: ModelGroup = { model_group: "custom-proxy-model" };
    expect(isModelCompatibleWithEndpoint(model, EndpointType.CHAT)).toBe(true);
    expect(isModelCompatibleWithEndpoint(model, EndpointType.REALTIME)).toBe(true);
    expect(isModelCompatibleWithEndpoint(model, EndpointType.SPEECH)).toBe(true);
  });

  it("keeps chat models for responses, anthropic messages, and interactions", () => {
    const chatModel: ModelGroup = { model_group: "gpt-4o", mode: "chat" };
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.RESPONSES)).toBe(true);
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.ANTHROPIC_MESSAGES)).toBe(true);
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.INTERACTIONS)).toBe(true);
    expect(isModelCompatibleWithEndpoint(chatModel, EndpointType.SPEECH)).toBe(false);
  });

  it("keeps image models for image_edits", () => {
    const imageModel: ModelGroup = { model_group: "dall-e-3", mode: "image_generation" };
    expect(isModelCompatibleWithEndpoint(imageModel, EndpointType.IMAGE_EDITS)).toBe(true);
    expect(isModelCompatibleWithEndpoint(imageModel, EndpointType.IMAGE)).toBe(true);
    expect(isModelCompatibleWithEndpoint(imageModel, EndpointType.CHAT)).toBe(false);
  });

  it("keeps only realtime models for the realtime endpoint", () => {
    const models: ModelGroup[] = [
      { model_group: "gpt-4o", mode: "chat" },
      { model_group: "gpt-realtime", mode: "realtime" },
      { model_group: "no-mode" },
    ];

    expect(filterModelsForEndpoint(models, EndpointType.REALTIME).map((m) => m.model_group)).toEqual([
      "gpt-realtime",
      "no-mode",
    ]);
  });

  it("excludes unknown modes from conversational endpoints", () => {
    const batchModel: ModelGroup = { model_group: "batch-job", mode: "batch" };
    const rerankModel: ModelGroup = { model_group: "reranker", mode: "rerank" };
    expect(isModelCompatibleWithEndpoint(batchModel, EndpointType.CHAT)).toBe(false);
    expect(isModelCompatibleWithEndpoint(rerankModel, EndpointType.RESPONSES)).toBe(false);
    expect(isModelCompatibleWithEndpoint(batchModel, EndpointType.REALTIME)).toBe(false);
  });

  it("keeps completion-mode models for the chat endpoint", () => {
    const completionModel: ModelGroup = { model_group: "davinci-002", mode: "completion" };
    expect(isModelCompatibleWithEndpoint(completionModel, EndpointType.CHAT)).toBe(true);
    expect(isModelCompatibleWithEndpoint(completionModel, EndpointType.RESPONSES)).toBe(true);
    expect(isModelCompatibleWithEndpoint(completionModel, EndpointType.SPEECH)).toBe(false);
  });

  it("keeps image-edit models for the image-edits endpoint using the mode the backend sends", () => {
    const imageEditModel: ModelGroup = { model_group: "gpt-image-1", mode: "image_edit" };
    const imageModel: ModelGroup = { model_group: "dall-e-3", mode: "image_generation" };

    expect(isModelCompatibleWithEndpoint(imageEditModel, EndpointType.IMAGE_EDITS)).toBe(true);
    expect(
      filterModelsForEndpoint([imageEditModel, imageModel], EndpointType.IMAGE_EDITS).map((m) => m.model_group),
    ).toEqual(["gpt-image-1", "dall-e-3"]);
  });
});
