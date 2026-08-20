import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import MCPToolArgumentsForm, { MCPToolArgumentsFormRef } from "./MCPToolArgumentsForm";
import { MCPTool, InputSchema } from "./types";

const toolWith = (schema: InputSchema | string): MCPTool =>
  ({ name: "demo_tool", description: "", inputSchema: schema, mcp_info: {} }) as unknown as MCPTool;

const renderForm = (schema: InputSchema | string) => {
  const ref = React.createRef<MCPToolArgumentsFormRef>();
  render(<MCPToolArgumentsForm ref={ref} tool={toolWith(schema)} />);
  return ref;
};

const submit = async (ref: React.RefObject<MCPToolArgumentsFormRef | null>) => ref.current!.getSubmitValues();

const submitError = async (ref: React.RefObject<MCPToolArgumentsFormRef | null>) => {
  try {
    await ref.current!.getSubmitValues();
    return null;
  } catch (error) {
    return error;
  }
};

describe("MCPToolArgumentsForm", () => {
  it("returns typed values for a string, integer, number and boolean field", async () => {
    const user = userEvent.setup();
    const ref = renderForm({
      type: "object",
      properties: {
        city: { type: "string" },
        count: { type: "integer" },
        ratio: { type: "number" },
        verbose: { type: "boolean" },
      },
      required: [],
    });

    await user.type(screen.getByPlaceholderText("Enter city"), "berlin");
    await user.clear(screen.getByPlaceholderText("Enter count"));
    await user.type(screen.getByPlaceholderText("Enter count"), "7");
    await user.clear(screen.getByPlaceholderText("Enter ratio"));
    await user.type(screen.getByPlaceholderText("Enter ratio"), "1.5");

    const expected = { city: "berlin", count: 7, ratio: 1.5, verbose: false };
    await expect(submit(ref)).resolves.toEqual(expected);
  });

  it("truncates a fractional value for an integer field", async () => {
    const user = userEvent.setup();
    const ref = renderForm({ type: "object", properties: { count: { type: "integer" } }, required: [] });

    await user.clear(screen.getByPlaceholderText("Enter count"));
    await user.type(screen.getByPlaceholderText("Enter count"), "9.8");

    await expect(submit(ref)).resolves.toEqual({ count: 9 });
  });

  it("drops an empty optional string rather than sending an empty value", async () => {
    const ref = renderForm({
      type: "object",
      properties: { city: { type: "string" }, country: { type: "string" } },
      required: [],
    });

    await expect(submit(ref)).resolves.toEqual({});
  });

  it("parses a JSON object field into an object", async () => {
    const user = userEvent.setup();
    const ref = renderForm({ type: "object", properties: { filters: { type: "object" } }, required: [] });

    const textarea = screen.getByPlaceholderText("Enter JSON object for filters");
    await user.clear(textarea);
    await user.type(textarea, '{{"a": 1}');

    await expect(submit(ref)).resolves.toEqual({ filters: { a: 1 } });
  });

  it("parses a JSON array field into an array", async () => {
    const user = userEvent.setup();
    const ref = renderForm({ type: "object", properties: { tags: { type: "array" } }, required: [] });

    const textarea = screen.getByPlaceholderText("Enter JSON array for tags");
    await user.clear(textarea);
    await user.type(textarea, '[["x","y"]');

    await expect(submit(ref)).resolves.toEqual({ tags: ["x", "y"] });
  });

  it("rejects and reports invalid JSON for an object field", async () => {
    const user = userEvent.setup();
    const ref = renderForm({ type: "object", properties: { filters: { type: "object" } }, required: [] });

    const textarea = screen.getByPlaceholderText("Enter JSON object for filters");
    await user.clear(textarea);
    await user.type(textarea, "not json");

    expect(await submitError(ref)).not.toBeNull();
    expect(await screen.findByText("Invalid JSON")).toBeInTheDocument();
  });

  it("rejects a JSON array typed into an object field", async () => {
    const user = userEvent.setup();
    const ref = renderForm({ type: "object", properties: { filters: { type: "object" } }, required: [] });

    const textarea = screen.getByPlaceholderText("Enter JSON object for filters");
    await user.clear(textarea);
    await user.type(textarea, "[[1,2]");

    expect(await submitError(ref)).not.toBeNull();
    expect(await screen.findByText("Please enter a JSON object")).toBeInTheDocument();
  });

  it("rejects an empty required field with the per-field message", async () => {
    const ref = renderForm({ type: "object", properties: { city: { type: "string" } }, required: ["city"] });

    expect(await submitError(ref)).not.toBeNull();
    expect(await screen.findByText("Please enter city")).toBeInTheDocument();
  });

  it("rejects with a non-Error carrying errorFields, which is what the caller branches on", async () => {
    const ref = renderForm({ type: "object", properties: { city: { type: "string" } }, required: ["city"] });

    const error = await submitError(ref);
    expect(error).not.toBeInstanceOf(Error);
    expect(error).toMatchObject({ errorFields: [{ name: ["city"], errors: ["Please enter city"] }] });
  });

  it("wraps values under params when the schema nests them", async () => {
    const user = userEvent.setup();
    const ref = renderForm({
      type: "object",
      properties: {
        params: { type: "object", properties: { city: { type: "string" } }, required: [] },
      },
      required: [],
    });

    await user.type(screen.getByPlaceholderText("Enter city"), "oslo");

    await expect(submit(ref)).resolves.toEqual({ params: { city: "oslo" } });
  });

  it("renders a single input field when the schema is only a string", async () => {
    const user = userEvent.setup();
    const ref = renderForm("tool_input_schema");

    await user.type(screen.getByPlaceholderText("Enter input for this tool"), "hello");

    await expect(submit(ref)).resolves.toEqual({ input: "hello" });
  });

  it("reports the required message for the string-schema input", async () => {
    const ref = renderForm("tool_input_schema");

    expect(await submitError(ref)).not.toBeNull();
    expect(await screen.findByText("Please enter input for this tool")).toBeInTheDocument();
  });

  it("seeds a schema default and sends it untouched", async () => {
    const ref = renderForm({
      type: "object",
      properties: { city: { type: "string", default: "paris" }, count: { type: "integer", default: 3 } },
      required: [],
    });

    await expect(submit(ref)).resolves.toEqual({ city: "paris", count: 3 });
  });

  it("shows the empty state and submits nothing when the schema has no properties", async () => {
    const ref = renderForm({ type: "object" } as InputSchema);

    expect(screen.getByText("No parameters required for this tool.")).toBeInTheDocument();
    await expect(submit(ref)).resolves.toEqual({});
  });
});
