import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-long-condition-chain.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module" },
});

ruleTester.run("no-long-condition-chain", rule as never, {
  valid: [
    "const x = a && b && c;",
    "const x = a || b || c;",
    "const x = a && (b || c);",
    "const x = a && b;",
    "if (a || b || c) {}",
    "const x = a ?? b ?? c;",
    "const url = a ?? b ?? c ?? d;",
    "const x = (a && b) ?? c ?? d;",
    { code: "const x = a && b && c && d;", options: [{ minConditions: 5 }] },
  ],
  invalid: [
    {
      code: "const x = a && b && c && d;",
      errors: [{ messageId: "tooMany", data: { count: 4 } }],
    },
    {
      code: "const x = a || b || c || d || e;",
      errors: [{ messageId: "tooMany", data: { count: 5 } }],
    },
    {
      code: "const x = a && b || c && d;",
      errors: [{ messageId: "tooMany", data: { count: 4 } }],
    },
    {
      code: "if (!a && !b && !c && !d) {}",
      errors: [{ messageId: "tooMany", data: { count: 4 } }],
    },
    {
      code: "const x = a && (b || c);",
      options: [{ minConditions: 3 }],
      errors: [{ messageId: "tooMany", data: { count: 3 } }],
    },
    {
      code: "const x = (a && b && c && d) || (e && f && g && h);",
      errors: [{ messageId: "tooMany", data: { count: 8 } }],
    },
    {
      code: "const x = (a && b && c && d) ?? fallback;",
      errors: [{ messageId: "tooMany", data: { count: 4 } }],
    },
  ],
});
