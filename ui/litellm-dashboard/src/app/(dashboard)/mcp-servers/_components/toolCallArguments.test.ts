import { describe, expect, it } from "vitest";

import { InputSchema } from "@/components/mcp_tools/types";
import {
  ToolArgumentField,
  buildToolCallArguments,
  hasNestedParamsSchema,
  toolArgumentFields,
  toolArgumentsResolver,
  validateToolArgument,
} from "./toolCallArguments";

const field = (
  key: string,
  type: string,
  required = false,
  extra: Record<string, unknown> = {},
): ToolArgumentField => ({
  key,
  prop: { type, ...extra },
  required,
});

describe("toolArgumentFields", () => {
  it("preserves schema property order and marks required entries", () => {
    const schema: InputSchema = {
      type: "object",
      properties: { b: { type: "string" }, a: { type: "integer" }, c: { type: "boolean" } },
      required: ["a"],
    };

    expect(toolArgumentFields(schema).map((f) => [f.key, f.required])).toEqual([
      ["b", false],
      ["a", true],
      ["c", false],
    ]);
  });

  it("returns no fields when the schema declares no properties", () => {
    expect(toolArgumentFields({ type: "object" } as InputSchema)).toEqual([]);
  });
});

describe("buildToolCallArguments", () => {
  it("keys the payload by schema key, not by field index", () => {
    const fields = [field("alpha", "string"), field("beta", "string")];

    expect(buildToolCallArguments(fields, ["one", "two"])).toEqual({ alpha: "one", beta: "two" });
  });

  it("keeps a dotted schema key flat instead of nesting it", () => {
    const result = buildToolCallArguments([field("filter.name", "string")], ["acme"]);

    expect(result).toEqual({ "filter.name": "acme" });
    expect(result).not.toHaveProperty("filter");
  });

  it("keeps a bracketed schema key flat instead of building an array", () => {
    const result = buildToolCallArguments([field("items[0]", "string")], ["x"]);

    expect(result).toEqual({ "items[0]": "x" });
    expect(result).not.toHaveProperty("items");
  });

  it("trims strings and drops fields that are blank after trimming", () => {
    const fields = [field("kept", "string"), field("blank", "string"), field("spaces", "string")];

    expect(buildToolCallArguments(fields, ["  hi  ", "", "   "])).toEqual({ kept: "hi" });
  });

  it("drops undefined and null values", () => {
    const fields = [field("a", "string"), field("b", "string")];

    expect(buildToolCallArguments(fields, [undefined, null])).toEqual({});
  });

  it("truncates integers and preserves floats", () => {
    const fields = [field("i", "integer"), field("n", "number")];

    expect(buildToolCallArguments(fields, ["7.9", "1.25"])).toEqual({ i: 7, n: 1.25 });
  });

  it("keeps a non-numeric string as-is on a numeric field", () => {
    expect(buildToolCallArguments([field("n", "number")], ["abc"])).toEqual({ n: "abc" });
  });

  it("emits real booleans from both the string and the boolean form", () => {
    const fields = [field("a", "boolean"), field("b", "boolean"), field("c", "boolean")];

    expect(buildToolCallArguments(fields, ["true", true, false])).toEqual({ a: true, b: true, c: false });
  });

  it("parses valid JSON objects and arrays", () => {
    const fields = [field("o", "object"), field("a", "array")];

    expect(buildToolCallArguments(fields, ['{"k":1}', "[1,2]"])).toEqual({ o: { k: 1 }, a: [1, 2] });
  });

  it("passes the raw string through when JSON is malformed or the wrong shape", () => {
    const fields = [field("o", "object"), field("a", "array")];

    expect(buildToolCallArguments(fields, ["not json", '{"k":1}'])).toEqual({ o: "not json", a: '{"k":1}' });
  });

  it("stringifies a numeric value declared as a string field", () => {
    expect(buildToolCallArguments([field("s", "string")], [42])).toEqual({ s: "42" });
  });
});

describe("validateToolArgument", () => {
  it("reports the required message for a blank required field", () => {
    expect(validateToolArgument(field("name", "string", true), "   ")).toBe("Please enter name");
  });

  it("accepts a populated required field", () => {
    expect(validateToolArgument(field("name", "string", true), "x")).toBeUndefined();
  });

  it("ignores an empty optional JSON field", () => {
    expect(validateToolArgument(field("o", "object"), "")).toBeUndefined();
  });

  it("reports malformed JSON", () => {
    expect(validateToolArgument(field("o", "object"), "{oops")).toBe("Invalid JSON");
  });

  it("reports an array supplied to an object field", () => {
    expect(validateToolArgument(field("o", "object"), "[1,2]")).toBe("Please enter a JSON object");
  });

  it("reports an object supplied to an array field", () => {
    expect(validateToolArgument(field("a", "array"), '{"k":1}')).toBe("Please enter a JSON array");
  });

  it("accepts well-formed values for both JSON field types", () => {
    expect(validateToolArgument(field("o", "object"), '{"k":1}')).toBeUndefined();
    expect(validateToolArgument(field("a", "array"), "[1]")).toBeUndefined();
  });
});

describe("toolArgumentsResolver", () => {
  it("returns the values untouched when every field is valid", () => {
    const fields = [field("a", "string", true), field("o", "object")];
    const values = { args: ["x", '{"k":1}'] };

    expect(toolArgumentsResolver(fields)(values, undefined, { fields: {}, shouldUseNativeValidation: false })).toEqual({
      values,
      errors: {},
    });
  });

  it("reports each failing field under its own index", async () => {
    const fields = [field("a", "string", true), field("o", "object"), field("b", "string")];

    const result = await toolArgumentsResolver(fields)({ args: ["", "{oops", "fine"] }, undefined, {
      fields: {},
      shouldUseNativeValidation: false,
    });

    expect(result.values).toEqual({});
    expect(result.errors).toEqual({
      args: {
        0: { type: "validate", message: "Please enter a" },
        1: { type: "validate", message: "Invalid JSON" },
      },
    });
  });
});

describe("hasNestedParamsSchema", () => {
  it("detects the nested params wrapper", () => {
    const schema: InputSchema = {
      type: "object",
      properties: { params: { type: "object", properties: { q: { type: "string" } } } },
    };

    expect(hasNestedParamsSchema(schema)).toBe(true);
  });

  it("rejects a params property that is not an object with properties", () => {
    expect(hasNestedParamsSchema({ type: "object", properties: { params: { type: "string" } } })).toBe(false);
    expect(hasNestedParamsSchema({ type: "object", properties: { params: { type: "object" } } })).toBe(false);
  });

  it("rejects a schema with no params property", () => {
    expect(hasNestedParamsSchema({ type: "object", properties: { q: { type: "string" } } })).toBe(false);
  });
});
