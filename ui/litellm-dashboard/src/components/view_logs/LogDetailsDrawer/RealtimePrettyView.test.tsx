import React from "react";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RealtimePrettyView, isRealtimeResponse } from "./RealtimePrettyView";

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
    },
  };
});

const sampleRealtimeResponse = {
  usage: {
    total_tokens: 587,
    prompt_tokens: 294,
    completion_tokens: 293,
  },
  results: [
    {
      type: "session.created",
      session: {
        id: "sess_DDNQlPKHjLsokSJPAOWY0",
        model: "gpt-4o-mini-realtime-preview",
        tools: [],
        voice: "alloy",
        modalities: ["audio", "text"],
        temperature: 0.8,
        tool_choice: "auto",
        instructions: "You are a helpful assistant.",
        turn_detection: {
          type: "server_vad",
          threshold: 0.5,
        },
        input_audio_format: "pcm16",
        output_audio_format: "pcm16",
        max_response_output_tokens: "inf",
      },
      event_id: "event_DDNQlB4VNUlpqTVIjBbm3",
    },
    {
      type: "response.done",
      event_id: "event_DDNQnagYJCZyZATdJCn0L",
      response: {
        id: "resp_DDNQnlXGHZJB46D5JhJ95",
        usage: {
          input_tokens: 116,
          total_tokens: 162,
          output_tokens: 46,
          input_token_details: {
            text_tokens: 116,
            audio_tokens: 0,
          },
          output_token_details: {
            text_tokens: 16,
            audio_tokens: 30,
          },
        },
        voice: "alloy",
        object: "realtime.response",
        output: [
          {
            id: "item_DDNQnz5uN1b8NEvPOPZOM",
            role: "assistant",
            type: "message",
            status: "completed",
            content: [
              {
                type: "audio",
                transcript: "Hello! How's your day going?",
              },
            ],
          },
        ],
        status: "completed",
        conversation_id: "conv_DDNQlpNllPYhCCfXCtT8X",
        max_output_tokens: "inf",
      },
    },
    {
      type: "response.done",
      event_id: "event_DDNR0VmrRTVU69RxGC29U",
      response: {
        id: "resp_DDNQy6S4PBZxW4qsKq6Ah",
        usage: {
          input_tokens: 178,
          total_tokens: 425,
          output_tokens: 247,
        },
        voice: "alloy",
        object: "realtime.response",
        output: [
          {
            id: "item_DDNQywctWVnYmujg4FSTZ",
            role: "assistant",
            type: "message",
            status: "completed",
            content: [
              {
                type: "audio",
                transcript:
                  "I'm here to help with information and general questions.",
              },
            ],
          },
        ],
        status: "completed",
        conversation_id: "conv_DDNQlpNllPYhCCfXCtT8X",
        max_output_tokens: "inf",
      },
    },
  ],
};

describe("isRealtimeResponse", () => {
  it("should return true for a valid realtime response with session.created", () => {
    expect(isRealtimeResponse(sampleRealtimeResponse)).toBe(true);
  });

  it("should return true for response with only response.done events", () => {
    const resp = {
      results: [{ type: "response.done", response: { id: "r1" } }],
    };
    expect(isRealtimeResponse(resp)).toBe(true);
  });

  it("should return false for a standard chat completion response", () => {
    const chatResponse = {
      choices: [{ message: { role: "assistant", content: "Hello" } }],
    };
    expect(isRealtimeResponse(chatResponse)).toBe(false);
  });

  it("should return false for null/undefined", () => {
    expect(isRealtimeResponse(null)).toBe(false);
    expect(isRealtimeResponse(undefined)).toBe(false);
  });

  it("should return false for empty results array", () => {
    expect(isRealtimeResponse({ results: [] })).toBe(false);
  });

  it("should return false for results with unrecognized event types", () => {
    const resp = {
      results: [{ type: "some.unknown.event" }],
    };
    expect(isRealtimeResponse(resp)).toBe(false);
  });
});

describe("RealtimePrettyView", () => {
  const mockWriteText = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: mockWriteText },
      writable: true,
      configurable: true,
    });
  });

  it("should render the component successfully", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("Session")).toBeInTheDocument();
  });

  it("should display the session model name", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    const modelElements = screen.getAllByText("gpt-4o-mini-realtime-preview");
    expect(modelElements.length).toBeGreaterThanOrEqual(1);
  });

  it("should display the session voice tag", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    const voiceElements = screen.getAllByText("alloy");
    expect(voiceElements.length).toBeGreaterThanOrEqual(1);
  });

  it("should display modality tags", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("audio")).toBeInTheDocument();
    expect(screen.getByText("text")).toBeInTheDocument();
  });

  it("should display the turn count in session header", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("2 turns")).toBeInTheDocument();
  });

  it("should display singular 'turn' for a single response event", () => {
    const singleTurnResponse = {
      results: [
        {
          type: "session.created",
          session: {
            id: "sess_1",
            model: "gpt-4o-mini-realtime-preview",
            voice: "alloy",
            modalities: ["audio"],
          },
        },
        {
          type: "response.done",
          response: {
            id: "r1",
            status: "completed",
            output: [
              {
                id: "item1",
                role: "assistant",
                type: "message",
                content: [{ type: "audio", transcript: "Hi!" }],
              },
            ],
          },
        },
      ],
    };
    render(<RealtimePrettyView response={singleTurnResponse} />);
    expect(screen.getByText("1 turn")).toBeInTheDocument();
  });

  it("should display the turn count in the output section header", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("Turns: 2")).toBeInTheDocument();
  });

  it("should display the Output section header", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("Output")).toBeInTheDocument();
  });

  it("should display transcript text from response turns", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(
      screen.getByText("Hello! How's your day going?")
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "I'm here to help with information and general questions."
      )
    ).toBeInTheDocument();
  });

  it("should display completed status tags for response turns", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    const completedTags = screen.getAllByText("completed");
    expect(completedTags.length).toBe(2);
  });

  it("should display token usage per turn", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText("116 in / 46 out tokens")).toBeInTheDocument();
    expect(screen.getByText("178 in / 247 out tokens")).toBeInTheDocument();
  });

  it("should expand session details when session header is clicked", async () => {
    const user = userEvent.setup();
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);

    await user.click(screen.getByText("Session"));

    await waitFor(() => {
      expect(screen.getByText("Temperature")).toBeInTheDocument();
    });
  });

  it("should display session instructions when expanded", async () => {
    const user = userEvent.setup();
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);

    await user.click(screen.getByText("Session"));

    await waitFor(() => {
      expect(screen.getByText("Instructions")).toBeInTheDocument();
      expect(
        screen.getByText("You are a helpful assistant.")
      ).toBeInTheDocument();
    });
  });

  it("should display session audio format when expanded", async () => {
    const user = userEvent.setup();
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);

    await user.click(screen.getByText("Session"));

    await waitFor(() => {
      expect(screen.getByText("Input Audio Format")).toBeInTheDocument();
      expect(screen.getAllByText("pcm16").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("should display ASSISTANT label for output messages", () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    const assistantLabels = screen.getAllByText("ASSISTANT");
    expect(assistantLabels.length).toBe(2);
  });

  it("should display fallback message when no recognized events exist", () => {
    const emptyResponse = {
      results: [{ type: "unknown.event" }],
    };
    render(<RealtimePrettyView response={emptyResponse} />);
    expect(
      screen.getByText("No recognized realtime events found")
    ).toBeInTheDocument();
  });

  it("should handle response with no output items gracefully", () => {
    const noOutputResponse = {
      results: [
        {
          type: "response.done",
          response: {
            id: "r1",
            status: "completed",
            output: [],
          },
        },
      ],
    };
    render(<RealtimePrettyView response={noOutputResponse} />);
    expect(screen.getByText("completed")).toBeInTheDocument();
  });

  it("should display metrics tokens when provided", () => {
    render(
      <RealtimePrettyView
        response={sampleRealtimeResponse}
        metrics={{ completion_tokens: 500, output_cost: 0.005 }}
      />
    );
    expect(screen.getByText(/Tokens: 500/)).toBeInTheDocument();
    expect(screen.getByText(/Cost: \$0\.005000/)).toBeInTheDocument();
  });

  it("should toggle output section collapse when header is clicked", async () => {
    const user = userEvent.setup();
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);

    const transcript = screen.getByText("Hello! How's your day going?");
    expect(transcript).toBeVisible();

    const outputHeader = screen.getByText("Output").closest("div");
    if (outputHeader) {
      await user.click(outputHeader);
      await waitFor(() => {
        expect(transcript).not.toBeVisible();
      });
    }
  });

  it("should display token breakdown tags when input_token_details are present", async () => {
    render(<RealtimePrettyView response={sampleRealtimeResponse} />);
    expect(screen.getByText(/Text Tokens: 116/)).toBeInTheDocument();
  });

  it("should handle text content type in addition to audio", () => {
    const textResponse = {
      results: [
        {
          type: "response.done",
          response: {
            id: "r1",
            status: "completed",
            output: [
              {
                id: "item1",
                role: "assistant",
                type: "message",
                content: [
                  {
                    type: "text",
                    text: "This is a text response",
                  },
                ],
              },
            ],
          },
        },
      ],
    };
    render(<RealtimePrettyView response={textResponse} />);
    expect(screen.getByText("This is a text response")).toBeInTheDocument();
  });
});
