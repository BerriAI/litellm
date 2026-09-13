import { describe, expect, it } from "vitest";

import { InputSchema, InputSchemaProperty } from "@/components/mcp_tools/types";
import {
  ToolArgumentField,
  argumentsFormKey,
  buildToolCallArguments,
  hasNestedParamsSchema,
  initialArgumentValues,
  resolveSchemaProperty,
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

describe("initialArgumentValues", () => {
  const seed = (schema: InputSchema): unknown[] => initialArgumentValues(toolArgumentFields(schema));

  it("seeds each primitive type with its empty value when the schema declares no default", () => {
    expect(
      seed({
        type: "object",
        properties: {
          message: { type: "string" },
          attempts: { type: "integer" },
          ratio: { type: "number" },
          active: { type: "boolean" },
        },
      }),
    ).toEqual(["", 0, 0, false]);
  });

  it("prefers a declared default over the empty value", () => {
    expect(
      seed({
        type: "object",
        properties: {
          label: { type: "string", default: "seeded" },
          ratio: { type: "number", default: 0.4 },
          active: { type: "boolean", default: true },
        },
      }),
    ).toEqual(["seeded", 0.4, true]);
  });

  it("seeds a false boolean default rather than falling back to the empty value", () => {
    expect(seed({ type: "object", properties: { active: { type: "boolean", default: false } } })).toEqual([false]);
  });

  it("renders an object field as pretty JSON built from its nested property defaults", () => {
    const [payload] = seed({
      type: "object",
      properties: {
        payload: {
          type: "object",
          properties: {
            id: { type: "string" },
            count: { type: "integer", default: 3 },
          },
        },
      },
    });

    expect(payload).toBe(JSON.stringify({ id: "", count: 3 }, null, 2));
  });

  it("keeps keys a declared object default carries that the schema does not describe", () => {
    const [payload] = seed({
      type: "object",
      properties: {
        payload: {
          type: "object",
          properties: { id: { type: "string" } },
          default: { id: "abc", extra: true },
        },
      },
    });

    expect(payload).toBe(JSON.stringify({ id: "abc", extra: true }, null, 2));
  });

  it("renders an array field as pretty JSON holding one sample item built from its item schema", () => {
    const [tags] = seed({
      type: "object",
      properties: { tags: { type: "array", items: { type: "string" } } },
    });

    expect(tags).toBe(JSON.stringify([""], null, 2));
  });

  it("rebuilds each entry of a declared array default against the item schema", () => {
    const [rows] = seed({
      type: "object",
      properties: {
        rows: {
          type: "array",
          items: { type: "object", properties: { id: { type: "string" }, n: { type: "integer" } } },
          default: [{ id: "a" }],
        },
      },
    });

    expect(rows).toBe(JSON.stringify([{ id: "a", n: 0 }], null, 2));
  });

  it("does not mutate the schema it seeds from", () => {
    const schema: InputSchema = {
      type: "object",
      properties: {
        payload: { type: "object", properties: { id: { type: "string" } }, default: { other: 1 } },
      },
    };
    const before = JSON.stringify(schema);

    seed(schema);

    expect(JSON.stringify(schema)).toBe(before);
  });

  it("seeds positionally, so a dotted key is just another index", () => {
    expect(
      seed({
        type: "object",
        properties: { "filter.name": { type: "string", default: "acme" }, plain: { type: "string" } },
      }),
    ).toEqual(["acme", ""]);
  });
});

describe("argumentsFormKey", () => {
  it("changes when a property's type changes under the same property names", () => {
    const before: InputSchema = { type: "object", properties: { value: { type: "string" } } };
    const after: InputSchema = { type: "object", properties: { value: { type: "integer" } } };

    expect(argumentsFormKey(after)).not.toBe(argumentsFormKey(before));
  });

  it("changes when a property's default changes", () => {
    const before: InputSchema = { type: "object", properties: { value: { type: "string", default: "a" } } };
    const after: InputSchema = { type: "object", properties: { value: { type: "string", default: "b" } } };

    expect(argumentsFormKey(after)).not.toBe(argumentsFormKey(before));
  });

  it("changes when a property is added", () => {
    const before: InputSchema = { type: "object", properties: { a: { type: "string" } } };
    const after: InputSchema = { type: "object", properties: { a: { type: "string" }, b: { type: "string" } } };

    expect(argumentsFormKey(after)).not.toBe(argumentsFormKey(before));
  });

  it("is stable across separately built but identical schemas", () => {
    const one: InputSchema = { type: "object", properties: { a: { type: "string", default: "x" } } };
    const two: InputSchema = { type: "object", properties: { a: { type: "string", default: "x" } } };

    expect(argumentsFormKey(one)).toBe(argumentsFormKey(two));
  });
});

describe("union-typed properties", () => {
  const optionalArray: InputSchemaProperty = {
    anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }],
    default: null,
  };
  const optionalObject: InputSchemaProperty = {
    anyOf: [{ type: "object", properties: { id: { type: "string" } } }, { type: "null" }],
    default: null,
  };
  const unionField = (key: string, prop: InputSchemaProperty, required = false): ToolArgumentField => ({
    key,
    prop,
    required,
  });

  describe("resolveSchemaProperty", () => {
    it("collapses an optional array to the array member", () => {
      expect(resolveSchemaProperty(optionalArray)).toMatchObject({ type: "array", items: { type: "string" } });
    });

    it("carries the outer description and default onto the resolved member", () => {
      const resolved = resolveSchemaProperty({
        anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }],
        description: "outer text",
        default: ["a"],
      });

      expect(resolved).toMatchObject({ type: "array", description: "outer text", default: ["a"] });
    });

    it("keeps the enum of a resolved string member so the select still renders", () => {
      expect(resolveSchemaProperty({ anyOf: [{ type: "string", enum: ["a", "b"] }, { type: "null" }] })).toMatchObject({
        type: "string",
        enum: ["a", "b"],
      });
    });

    it("resolves oneOf the same way as anyOf", () => {
      expect(resolveSchemaProperty({ oneOf: [{ type: "object" }, { type: "null" }] })).toMatchObject({
        type: "object",
      });
    });

    it("leaves a property that already declares a type untouched", () => {
      const plain: InputSchemaProperty = { type: "string", anyOf: [{ type: "array" }, { type: "null" }] };

      expect(resolveSchemaProperty(plain)).toBe(plain);
    });

    it("leaves a genuine multi-type union unresolved", () => {
      const multi: InputSchemaProperty = { anyOf: [{ type: "string" }, { type: "integer" }, { type: "null" }] };

      expect(resolveSchemaProperty(multi)).toBe(multi);
    });
  });

  describe("validateToolArgument", () => {
    it("reports an object supplied to an optional array parameter", () => {
      expect(validateToolArgument(unionField("tags", optionalArray), '{"k":1}')).toBe("Please enter a JSON array");
    });

    it("reports an array supplied to an optional object parameter", () => {
      expect(validateToolArgument(unionField("payload", optionalObject), "[1,2]")).toBe("Please enter a JSON object");
    });

    it("reports the comma-separated text a plain input would have produced as invalid JSON", () => {
      expect(validateToolArgument(unionField("tags", optionalArray), "a,b")).toBe("Invalid JSON");
    });

    it("accepts a well-formed value and an empty optional value", () => {
      expect(validateToolArgument(unionField("tags", optionalArray), '["a","b"]')).toBeUndefined();
      expect(validateToolArgument(unionField("tags", optionalArray), "")).toBeUndefined();
    });
  });

  describe("buildToolCallArguments", () => {
    it("sends a real array for an optional array parameter", () => {
      expect(buildToolCallArguments([unionField("tags", optionalArray)], ['["a","b"]'])).toEqual({
        tags: ["a", "b"],
      });
    });

    it("sends a real object for an optional object parameter", () => {
      expect(buildToolCallArguments([unionField("payload", optionalObject)], ['{"id":"x"}'])).toEqual({
        payload: { id: "x" },
      });
    });
  });

  describe("initialArgumentValues", () => {
    it("leaves a null-defaulted optional parameter blank so it is not submitted at all", () => {
      const fields = [unionField("tags", optionalArray), unionField("payload", optionalObject)];

      expect(initialArgumentValues(fields)).toEqual(["", ""]);
      expect(buildToolCallArguments(fields, initialArgumentValues(fields))).toEqual({});
    });

    it("builds a sample item from the resolved item schema when the union declares no default", () => {
      const prop: InputSchemaProperty = { anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }] };

      expect(initialArgumentValues([unionField("tags", prop)])).toEqual([JSON.stringify([""], null, 2)]);
    });

    it("seeds a nested optional array inside an object from the resolved member", () => {
      const [payload] = initialArgumentValues([
        unionField("payload", {
          type: "object",
          properties: { tags: { anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }] } },
        }),
      ]);

      expect(payload).toBe(JSON.stringify({ tags: [""] }, null, 2));
    });
  });
});
