/**
 * Core tests for Tools section
 */

import { describe, it, expect } from "vitest";
import { parseToolsFromLog } from "./utils";
import { LogEntry } from "../columns";

describe("ToolsSection", () => {
  it("should parse tools from request and match with response tool calls", () => {
    const mockLog: LogEntry = {
      request_id: "test-123",
      api_key: "key",
      team_id: "team",
      model: "gpt-4",
      model_id: "gpt-4",
      call_type: "completion",
      spend: 0.01,
      total_tokens: 100,
      prompt_tokens: 50,
      completion_tokens: 50,
      startTime: "2024-01-01T00:00:00Z",
      endTime: "2024-01-01T00:00:01Z",
      cache_hit: "none",
      messages: JSON.stringify({
        model: "gpt-4",
        messages: [{ role: "user", content: "What's the weather?" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get the current weather",
              parameters: {
                type: "object",
                required: ["location"],
                properties: {
                  location: { type: "string", description: "City name" },
                },
              },
            },
          },
          {
            type: "function",
            function: {
              name: "search_web",
              description: "Search the web",
              parameters: {
                type: "object",
                required: ["query"],
                properties: {
                  query: { type: "string", description: "Search query" },
                },
              },
            },
          },
        ],
      }),
      response: JSON.stringify({
        choices: [
          {
            message: {
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: '{"location": "San Francisco"}',
                  },
                },
              ],
            },
          },
        ],
      }),
    };

    const tools = parseToolsFromLog(mockLog);

    expect(tools).toHaveLength(2);
    expect(tools[0].name).toBe("get_weather");
    expect(tools[0].called).toBe(true);
    expect(tools[0].callData?.arguments).toEqual({ location: "San Francisco" });
    expect(tools[1].name).toBe("search_web");
    expect(tools[1].called).toBe(false);
  });

  it("should return empty array when no tools in request", () => {
    const mockLog: LogEntry = {
      request_id: "test-456",
      api_key: "key",
      team_id: "team",
      model: "gpt-4",
      model_id: "gpt-4",
      call_type: "completion",
      spend: 0.01,
      total_tokens: 100,
      prompt_tokens: 50,
      completion_tokens: 50,
      startTime: "2024-01-01T00:00:00Z",
      endTime: "2024-01-01T00:00:01Z",
      cache_hit: "none",
      messages: JSON.stringify({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      }),
      response: JSON.stringify({
        choices: [{ message: { content: "Hi there!" } }],
      }),
    };

    const tools = parseToolsFromLog(mockLog);

    expect(tools).toHaveLength(0);
  });
});
