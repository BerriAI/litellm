import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-large-inline-object-arg.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module" },
});

ruleTester.run("no-large-inline-object-arg", rule as never, {
  valid: [
    "foo({ a: 1, b: 2, c: 3 });",
    "foo({});",
    "const opts = { a: 1, b: 2, c: 3, d: 4 }; foo(opts);",
    "const x = { a: 1, b: 2, c: 3, d: 4 };",
    "function f() { return { a: 1, b: 2, c: 3, d: 4 }; }",
    "const arr = [{ a: 1, b: 2, c: 3, d: 4 }];",
    "foo(1, 2, { a: 1, b: 2 });",
    { code: "foo({ a: 1, b: 2, c: 3, d: 4 });", options: [{ minProperties: 5 }] },
  ],
  invalid: [
    {
      code: "foo({ a: 1, b: 2, c: 3, d: 4 });",
      errors: [{ messageId: "tooLarge", data: { count: 4 } }],
    },
    {
      code: "new Widget({ a: 1, b: 2, c: 3, d: 4, e: 5 });",
      errors: [{ messageId: "tooLarge", data: { count: 5 } }],
    },
    {
      code: "foo(1, { a: 1, b: 2, c: 3, d: 4 });",
      errors: [{ messageId: "tooLarge" }],
    },
    {
      code: "foo({ a: 1, ...rest, c: 3, d: 4 });",
      errors: [{ messageId: "tooLarge", data: { count: 4 } }],
    },
    {
      code: "foo({ a: 1, b: 2, c: 3 });",
      options: [{ minProperties: 3 }],
      errors: [{ messageId: "tooLarge", data: { count: 3 } }],
    },
    {
      code: "outer({ a: 1, b: 2, c: 3, d: 4 }, inner({ e: 5, f: 6, g: 7, h: 8 }));",
      errors: [{ messageId: "tooLarge" }, { messageId: "tooLarge" }],
    },
  ],
});
