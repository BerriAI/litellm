import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UserEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ToolTestPanel } from "./ToolTestPanel";
import { InputSchema, MCPTool } from "@/components/mcp_tools/types";

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
    expect(screen.getByTitle("True")).toBeInTheDocument();

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

  it("renders the call button as type=button so a click never also triggers native form submission", () => {
    const schema: InputSchema = {
      type: "object",
      properties: {
        message: { type: "string", description: "Prompt text" },
      },
    };

    renderPanel(schema);

    const callButton = screen.getByRole("button", { name: "Call Tool" });
    expect(callButton.closest("form")).not.toBeNull();
    expect(callButton).toHaveAttribute("type", "button");
  });
});

describe("ToolTestPanel argument payload", () => {
  const submitPanel = async (schema: InputSchema | string, drive?: (user: UserEvent) => Promise<void>) => {
    const onSubmit = vi.fn();
    render(
      <ToolTestPanel
        tool={buildTool(schema)}
        onSubmit={onSubmit}
        isLoading={false}
        result={null}
        error={null}
        onClose={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    if (drive) {
      await drive(user);
    }
    await user.click(screen.getByRole("button", { name: "Call Tool" }));
    return onSubmit;
  };

  it("sends a typed string under its own key, whitespace trimmed", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { message: { type: "string" } } },
      async (user) => {
        await user.type(screen.getByLabelText("message"), "  hello world  ");
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ message: "hello world" });
  });

  it("sends what the user typed into the fallback input when the tool has no real schema", async () => {
    const onSubmit = vi.fn();
    render(
      <ToolTestPanel
        tool={buildTool("tool_input_schema")}
        onSubmit={onSubmit}
        isLoading={false}
        result={null}
        error={null}
        onClose={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText("Enter input for this tool"), "  do the thing  ");
    await user.click(screen.getByRole("button", { name: "Call Tool" }));

    expect(onSubmit).toHaveBeenCalledWith({ input: "do the thing" });
  });

  it("coerces integer fields to truncated numbers and number fields to floats", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { attempts: { type: "integer" }, ratio: { type: "number" } } },
      async (user) => {
        await user.clear(screen.getByLabelText("attempts"));
        await user.type(screen.getByLabelText("attempts"), "7.9");
        await user.clear(screen.getByLabelText("ratio"));
        await user.type(screen.getByLabelText("ratio"), "1.25");
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ attempts: 7, ratio: 1.25 });
  });

  it("sends a real boolean, not the string 'true', when the boolean select is changed", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { active: { type: "boolean", default: false } } },
      async (user) => {
        await user.click(screen.getByLabelText("active"));
        await user.click(await screen.findByText("True"));
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ active: true });
  });

  it("parses object and array textareas into real JSON values", async () => {
    const onSubmit = await submitPanel(
      {
        type: "object",
        properties: { payload: { type: "object" }, tags: { type: "array" } },
      },
      async (user) => {
        await user.clear(screen.getByTestId("textarea-payload"));
        await user.type(screen.getByTestId("textarea-payload"), '{{"a":1}');
        await user.clear(screen.getByTestId("textarea-tags"));
        await user.type(screen.getByTestId("textarea-tags"), '[["x","y"]');
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ payload: { a: 1 }, tags: ["x", "y"] });
  });

  it("omits a field whose value is left empty", async () => {
    const onSubmit = await submitPanel({
      type: "object",
      properties: { message: { type: "string" }, note: { type: "string" } },
    });

    expect(onSubmit).toHaveBeenCalledWith({});
  });

  it("keeps a dotted schema key flat rather than nesting it", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { "filter.name": { type: "string" } } },
      async (user) => {
        await user.type(screen.getByLabelText("filter.name"), "acme");
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ "filter.name": "acme" });
    expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("filter");
  });

  it("sends the option picked from an enum select", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { mode: { type: "string", enum: ["fast", "thorough"] } } },
      async (user) => {
        await user.selectOptions(screen.getByLabelText("mode"), "thorough");
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ mode: "thorough" });
  });

  it("wraps the arguments back under params for a nested params schema", async () => {
    const onSubmit = await submitPanel(
      {
        type: "object",
        properties: {
          params: { type: "object", properties: { query: { type: "string" } } },
        },
      },
      async (user) => {
        await user.type(screen.getByLabelText("query"), "widgets");
      },
    );

    expect(onSubmit).toHaveBeenCalledWith({ params: { query: "widgets" } });
  });

  it("blocks the call and shows the required message when a required field is empty", async () => {
    const onSubmit = await submitPanel({
      type: "object",
      properties: { message: { type: "string" } },
      required: ["message"],
    });

    expect(await screen.findByText("Please enter message")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("blocks the call and shows the JSON message when an object field holds invalid JSON", async () => {
    const onSubmit = await submitPanel(
      { type: "object", properties: { payload: { type: "object" } } },
      async (user) => {
        await user.clear(screen.getByTestId("textarea-payload"));
        await user.type(screen.getByTestId("textarea-payload"), "not json");
      },
    );

    expect(await screen.findByText("Invalid JSON")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("sends the seeded defaults when the user submits without touching anything", async () => {
    const onSubmit = await submitPanel({
      type: "object",
      properties: {
        ratio: { type: "number", default: 0.4 },
        active: { type: "boolean", default: true },
        label: { type: "string", default: "seeded" },
      },
    });

    expect(onSubmit).toHaveBeenCalledWith({ ratio: 0.4, active: true, label: "seeded" });
  });
});

describe("ToolTestPanel schema changes under a stable tool name", () => {
  const renderWith = (schema: InputSchema, onSubmit: ReturnType<typeof vi.fn>) => (
    <ToolTestPanel
      tool={buildTool(schema)}
      onSubmit={onSubmit}
      isLoading={false}
      result={null}
      error={null}
      onClose={vi.fn()}
    />
  );

  it("reseeds the fields when the same-named tool's schema changes", async () => {
    const onSubmit = vi.fn();
    const before: InputSchema = { type: "object", properties: { message: { type: "string" } } };
    const after: InputSchema = {
      type: "object",
      properties: { query: { type: "string" }, limit: { type: "integer", default: 5 } },
    };

    const { rerender } = render(renderWith(before, onSubmit));
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("message"), "stale value");

    rerender(renderWith(after, onSubmit));

    expect(screen.queryByLabelText("message")).not.toBeInTheDocument();
    expect(screen.getByLabelText("query")).toHaveValue("");
    expect(screen.getByLabelText("limit")).toHaveValue(5);

    await user.type(screen.getByLabelText("query"), "widgets");
    await user.click(screen.getByRole("button", { name: "Call Tool" }));

    expect(onSubmit).toHaveBeenCalledWith({ query: "widgets", limit: 5 });
  });

  it("keeps what the user typed when the schema is rebuilt with identical content", async () => {
    const onSubmit = vi.fn();
    const schema = (): InputSchema => ({ type: "object", properties: { message: { type: "string" } } });

    const { rerender } = render(renderWith(schema(), onSubmit));
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("message"), "typed by hand");

    rerender(renderWith(schema(), onSubmit));

    expect(screen.getByLabelText("message")).toHaveValue("typed by hand");
  });
});

describe("ToolTestPanel optional union-typed parameters", () => {
  const qaEchoSchema: InputSchema = {
    type: "object",
    properties: {
      message: { type: "string" },
      repeat: { type: "integer", default: 1 },
      loud: { type: "boolean", default: false },
      tags: { anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }], default: null },
    },
    required: ["message"],
  };

  const runPanel = async (drive: () => void) => {
    const onSubmit = vi.fn();
    render(
      <ToolTestPanel
        tool={buildTool(qaEchoSchema)}
        onSubmit={onSubmit}
        isLoading={false}
        result={null}
        error={null}
        onClose={vi.fn()}
      />,
    );
    drive();
    await userEvent.setup().click(screen.getByRole("button", { name: "Call Tool" }));
    return onSubmit;
  };

  it("renders the JSON textarea for an optional array parameter, not a plain text input", () => {
    renderPanel(qaEchoSchema);

    const tags = screen.getByTestId("textarea-tags");
    expect(tags.tagName).toBe("TEXTAREA");
    expect(tags).toHaveValue("");
    expect(screen.getByPlaceholderText("Enter JSON array for tags")).toBe(tags);
    expect(screen.queryByPlaceholderText("Enter tags")).not.toBeInTheDocument();
  });

  it("renders the JSON textarea for an optional object parameter, not a plain text input", () => {
    renderPanel({
      type: "object",
      properties: {
        payload: { anyOf: [{ type: "object", properties: { id: { type: "string" } } }, { type: "null" }] },
      },
    });

    const payload = screen.getByTestId("textarea-payload");
    expect(payload).toHaveValue(JSON.stringify({ id: "" }, null, 2));
    expect(screen.getByPlaceholderText("Enter JSON object for payload")).toBe(payload);
    expect(screen.queryByPlaceholderText("Enter payload")).not.toBeInTheDocument();
  });

  it("sends an optional array parameter as a real array", async () => {
    const onSubmit = await runPanel(() => {
      fireEvent.change(screen.getByPlaceholderText("Enter message"), { target: { value: "hi" } });
      fireEvent.change(screen.getByTestId("textarea-tags"), { target: { value: '["a","b"]' } });
    });

    const expected = { message: "hi", repeat: 1, loud: false, tags: ["a", "b"] };
    expect(onSubmit).toHaveBeenCalledWith(expected);
  });

  it("omits an optional array parameter the user never filled in", async () => {
    const onSubmit = await runPanel(() => {
      fireEvent.change(screen.getByPlaceholderText("Enter message"), { target: { value: "hi" } });
    });

    expect(onSubmit).toHaveBeenCalledWith({ message: "hi", repeat: 1, loud: false });
  });

  it("blocks the call instead of sending comma-separated text as a raw string", async () => {
    const onSubmit = await runPanel(() => {
      fireEvent.change(screen.getByPlaceholderText("Enter message"), { target: { value: "hi" } });
      fireEvent.change(screen.getByTestId("textarea-tags"), { target: { value: "a,b" } });
    });

    expect(await screen.findByText("Invalid JSON")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
