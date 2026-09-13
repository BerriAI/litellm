import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PrettyMessagesView } from "./PrettyMessagesView";

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

  it("renders input when request is a bare messages array (cold storage payload)", () => {
    const request = [{ role: "user", content: "Write me a poem" }];
    const response = {
      choices: [{ message: { role: "assistant", content: "A quiet moment." } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Write me a poem")).toBeInTheDocument();
    expect(screen.getByText("A quiet moment.")).toBeInTheDocument();
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

  it("renders a Responses API log, whose body uses input/output instead of messages/choices", () => {
    const request = {
      model: "gpt-5.6",
      input: [{ role: "user", content: "Reply with exactly: hello from responses api" }],
    };
    const response = {
      output: [
        {
          id: "msg_070989277645d4ae",
          role: "assistant",
          type: "message",
          status: "completed",
          content: [{ text: "hello from responses api", type: "output_text", annotations: [] }],
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Reply with exactly: hello from responses api")).toBeInTheDocument();
    expect(screen.getByText("hello from responses api")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders a Responses API tool call, whose output item is a function_call", () => {
    const request = {
      model: "gpt-5.6",
      input: [{ role: "user", content: "What is the weather in San Francisco? Use the tool." }],
    };
    const response = {
      output: [
        {
          id: "fc_08edf6c2312f1485",
          name: "get_weather",
          type: "function_call",
          status: "completed",
          call_id: "call_AtO0J9eNy5jgECXzBicMJM8W",
          arguments: '{"city":"San Francisco"}',
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("What is the weather in San Francisco? Use the tool.")).toBeInTheDocument();
    expect(screen.getByText("get_weather")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders instructions as the system turn and a bare string input", () => {
    const request = { model: "gpt-5.6", instructions: "You are terse.", input: "Say A" };
    const response = {
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "A" }] }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("You are terse.")).toBeInTheDocument();
    expect(screen.getByText("Say A")).toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
  });

  it("skips reasoning output items rather than rendering them as empty turns", () => {
    const request = { input: [{ role: "user", content: "Think then answer" }] };
    const response = {
      output: [
        { type: "reasoning", id: "rs_1", summary: [] },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "answered" }] },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("answered")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders a Responses API follow-up turn carrying a prior function_call and its output", () => {
    const request = {
      input: [
        { role: "user", content: "What is the weather in San Francisco? Use the tool." },
        {
          type: "function_call",
          name: "get_weather",
          call_id: "call_AtO0J9eNy5jgECXzBicMJM8W",
          arguments: '{"city":"San Francisco"}',
        },
        { type: "function_call_output", call_id: "call_AtO0J9eNy5jgECXzBicMJM8W", output: '{"temp":18}' },
      ],
    };
    const response = {
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "It is 18 degrees." }] }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("It is 18 degrees.")).toBeInTheDocument();
    expect(screen.getByText('{"temp":18}')).toBeInTheDocument();
    expect(screen.getByText("TOOL")).toBeInTheDocument();
  });

  it("maps the developer and legacy function roles onto the roles the drawer renders", () => {
    const request = {
      messages: [
        { role: "developer", content: "Stay terse." },
        { role: "user", content: "Weather?" },
        { role: "function", name: "get_weather", content: '{"temp":18}' },
      ],
    };
    const response = { choices: [{ message: { role: "assistant", content: "18 degrees." } }] };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Stay terse.")).toBeInTheDocument();
    expect(screen.getByText("TOOL")).toBeInTheDocument();
    expect(screen.queryByText("FUNCTION")).not.toBeInTheDocument();
  });

  it("still reports missing output when a Responses API log has an empty output array", () => {
    const request = { input: [{ role: "user", content: "Hello" }] };

    render(<PrettyMessagesView request={request} response={{ output: [] }} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("No response data available")).toBeInTheDocument();
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
