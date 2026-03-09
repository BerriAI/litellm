import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PrettyMessagesView } from "./PrettyMessagesView";

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
    },
  };
});

describe("PrettyMessagesView", () => {
  it("should render the component for standard chat completions", () => {
    const request = {
      messages: [{ role: "user", content: "Hello" }],
    };
    const response = {
      choices: [{ message: { role: "assistant", content: "Hi there!" } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should render the realtime pretty view for realtime API responses", () => {
    const request = {};
    const response = {
      results: [
        {
          type: "session.created",
          session: {
            id: "sess_123",
            model: "gpt-4o-mini-realtime-preview",
            voice: "alloy",
            modalities: ["audio", "text"],
          },
        },
        {
          type: "response.done",
          response: {
            id: "resp_1",
            status: "completed",
            output: [
              {
                id: "item_1",
                role: "assistant",
                type: "message",
                content: [{ type: "audio", transcript: "Hello from realtime!" }],
              },
            ],
          },
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Session")).toBeInTheDocument();
    expect(screen.getByText("Hello from realtime!")).toBeInTheDocument();
    const modelElements = screen.getAllByText("gpt-4o-mini-realtime-preview");
    expect(modelElements.length).toBeGreaterThanOrEqual(1);
  });

  it("should render standard view when response has results but no realtime events", () => {
    const request = {
      messages: [{ role: "user", content: "Test" }],
    };
    const response = {
      results: [{ type: "some.other.type" }],
      choices: [{ message: { role: "assistant", content: "Reply" } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Test")).toBeInTheDocument();
    expect(screen.getByText("Reply")).toBeInTheDocument();
  });
});
