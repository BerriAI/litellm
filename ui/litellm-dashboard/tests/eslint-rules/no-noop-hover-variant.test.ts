import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-noop-hover-variant.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module" },
});

ruleTester.run("no-noop-hover-variant", rule as never, {
  valid: [
    'const c = "text-info hover:text-info/80";',
    'const c = "bg-success hover:bg-success/80 text-white";',
    'const c = "border-border hover:border-ring";',
    'const c = "text-info hover:underline";',
    'const c = "bg-info/10 hover:bg-info/20";',
    'const c = "hover:bg-muted";',
    'const c = "text-info focus:text-info";',
    "const c = `cursor-pointer hover:bg-info/10 ${isSelected ? 'bg-info/10' : ''}`;",
    "const c = `border-2 ${isSelected ? 'border-info bg-info/10' : 'border-border hover:border-ring'}`;",
  ],
  invalid: [
    {
      code: 'const c = "text-info hover:text-info";',
      errors: [{ messageId: "noop", data: { hover: "hover:text-info", base: "text-info" } }],
    },
    {
      code: 'const c = "text-xs bg-success hover:bg-success text-white transition-colors";',
      errors: [{ messageId: "noop", data: { hover: "hover:bg-success", base: "bg-success" } }],
    },
    {
      code: 'const c = "border border-border hover:border-border";',
      errors: [{ messageId: "noop", data: { hover: "hover:border-border", base: "border-border" } }],
    },
    {
      code: 'const c = "text-destructive border-destructive hover:text-destructive hover:border-destructive";',
      errors: [{ messageId: "noop" }, { messageId: "noop" }],
    },
    {
      code: 'const c = "text-info underline hover:underline";',
      errors: [{ messageId: "noop", data: { hover: "hover:underline", base: "underline" } }],
    },
    {
      code: "const c = `text-info hover:text-info ${extra}`;",
      errors: [{ messageId: "noop" }],
    },
    {
      code: "const c = `p-2 ${active ? 'bg-info/15 text-info' : 'text-success hover:text-success'}`;",
      errors: [{ messageId: "noop", data: { hover: "hover:text-success", base: "text-success" } }],
    },
  ],
});
