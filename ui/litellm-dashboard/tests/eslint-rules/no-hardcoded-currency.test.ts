import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-hardcoded-currency.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module", parserOptions: { ecmaFeatures: { jsx: true } } },
});

ruleTester.run("no-hardcoded-currency", rule as never, {
  valid: [
    'const regex = "^name$";',
    'const variable = "$HOME";',
    "const text = `${value} tokens`;",
    "const el = <span>Price in USD</span>;",
  ],
  invalid: [
    { code: 'const symbol = "$";', errors: [{ messageId: "hardcoded" }] },
    { code: 'const price = "$0.00";', errors: [{ messageId: "hardcoded" }] },
    { code: 'const small = "< $";', errors: [{ messageId: "hardcoded" }] },
    { code: "const price = `$${value}`;", errors: [{ messageId: "hardcoded" }] },
    { code: "const el = <span>${value}</span>;", errors: [{ messageId: "hardcoded" }] },
  ],
});
