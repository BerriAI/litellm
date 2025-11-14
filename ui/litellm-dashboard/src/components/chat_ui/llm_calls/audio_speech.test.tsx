import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIAudioSpeechRequest } from "./audio_speech";
import OpenAI from "openai";

vi.mock("openai");

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn(() => "blob:mock-audio-url");

describe("audio_speech", () => {
  const mockCreate = vi.fn();
  const mockUpdateUI = vi.fn();
  const mockBlob = new Blob(["mock audio data"], { type: "audio/mpeg" });
  let abortController: AbortController | null = null;

  beforeEach(() => {
    // Mock the response structure from OpenAI audio speech API
    mockCreate.mockResolvedValue({
      blob: vi.fn().mockResolvedValue(mockBlob),
    });

    // Mock the OpenAI constructor and its methods
    (OpenAI as any).mockImplementation(() => ({
      audio: {
        speech: {
          create: mockCreate,
        },
      },
    }));
  });

  afterEach(() => {
    // Clean up abort controller if it exists
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    vi.clearAllMocks();
  });

  it("should make a request to the audio speech API with basic parameters", async () => {
    await makeOpenAIAudioSpeechRequest("Hello, world!", "alloy", mockUpdateUI, "tts-1", "sk-1234567890", []);

    expect(mockCreate).toHaveBeenCalledWith(
      {
        model: "tts-1",
        input: "Hello, world!",
        voice: "alloy",
      },
      { signal: undefined },
    );
    expect(mockUpdateUI).toHaveBeenCalledWith("blob:mock-audio-url", "tts-1");
  });

  it("should include optional parameters when provided", async () => {
    abortController = new AbortController();
    const signal = abortController.signal;

    await makeOpenAIAudioSpeechRequest(
      "Test input",
      "nova",
      mockUpdateUI,
      "tts-1-hd",
      "sk-1234567890",
      ["tag1", "tag2"],
      signal,
      "mp3",
      1.5,
    );

    expect(mockCreate).toHaveBeenCalledWith(
      {
        model: "tts-1-hd",
        input: "Test input",
        voice: "nova",
        response_format: "mp3",
        speed: 1.5,
      },
      { signal },
    );
    expect(mockUpdateUI).toHaveBeenCalledWith("blob:mock-audio-url", "tts-1-hd");
  });

  it("should handle errors gracefully", async () => {
    const mockError = new Error("API Error");
    mockCreate.mockRejectedValue(mockError);

    await expect(
      makeOpenAIAudioSpeechRequest("Hello, world!", "alloy", mockUpdateUI, "tts-1", "sk-1234567890", []),
    ).rejects.toThrow("API Error");

    expect(mockUpdateUI).not.toHaveBeenCalled();
  });
});
