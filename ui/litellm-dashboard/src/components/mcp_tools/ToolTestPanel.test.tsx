import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ToolTestPanel } from "./ToolTestPanel";
import { InputSchema, MCPTool } from "./types";

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
}));

const buildTool = (schema: InputSchema | string): MCPTool => ({
  name: "demo-tool",
  description: "demo",
  inputSchema: schema,
  mcp_info: { server_name: "demo-server" },
});

const renderPanel = (schema: InputSchema | string) =>
  render(
    <ToolTestPanel
      tool={buildTool(schema)}
      onSubmit={vi.fn()}
      isLoading={false}
      result={null}
      error={null}
      onClose={vi.fn()}
    />,
  );

describe("ToolTestPanel defaults", () => {
  it("pre-populates primitive, array, and nested object inputs from schema", () => {
    const schema: InputSchema = {
      type: "object",
      properties: {
        message: { type: "string", description: "Prompt text" },
        attempts: { type: "integer" },
        ratio: { type: "number", default: 0.4 },
        active: { type: "boolean", default: true },
        keywords: {
          type: "array",
          items: { type: "string" },
          description: "keywords array",
        },
        payload: {
          type: "object",
          properties: {
            user: {
              type: "object",
              properties: {
                id: { type: "string", description: "user id" },
                tags: {
                  type: "array",
                  items: { type: "string" },
                  default: [],
                  description: "optional tags",
                },
              },
              required: ["id"],
            },
            context: {
              type: "object",
              properties: {
                topic: { type: "string" },
                extra: {
                  type: "object",
                  properties: {
                    note: { type: "string" },
                    score: { type: "number" },
                  },
                },
              },
              required: ["topic"],
            },
          },
          required: ["user", "context"],
        },
      },
    };

    renderPanel(schema);

    expect(screen.getByLabelText("message")).toHaveValue("");
    expect(screen.getByLabelText("attempts")).toHaveValue(0);
    expect(screen.getByLabelText("ratio")).toHaveValue(0.4);
    expect(screen.getByDisplayValue("True")).toBeInTheDocument();

    const keywordsTextarea = screen.getByTestId("textarea-keywords");
    expect(JSON.parse(keywordsTextarea.value)).toEqual([""]);

    const payloadTextarea = screen.getByTestId("textarea-payload");
    expect(JSON.parse(payloadTextarea.value)).toEqual({
      user: {
        id: "",
        tags: [""],
      },
      context: {
        topic: "",
        extra: {
          note: "",
          score: 0,
        },
      },
    });
  });

  it("uses nested params schema when present", () => {
    const schema: InputSchema = {
      type: "object",
      properties: {
        params: {
          type: "object",
          properties: {
            query: { type: "string" },
            filters: {
              type: "object",
              properties: {
                tag: { type: "string" },
                metadata: {
                  type: "object",
                  properties: {
                    source: { type: "string" },
                  },
                },
              },
            },
          },
        },
      },
    };

    renderPanel(schema);

    expect(screen.getByLabelText("query")).toBeInTheDocument();
    const filtersTextarea = screen.getByTestId("textarea-filters");
    expect(JSON.parse(filtersTextarea.value)).toEqual({
      tag: "",
      metadata: { source: "" },
    });
  });

  it("falls back to a plain input when schema is missing", () => {
    renderPanel("tool_input_schema");

    expect(screen.getByPlaceholderText("Enter input for this tool")).toBeInTheDocument();
    expect(screen.queryByText("No parameters required")).not.toBeInTheDocument();
  });
});
