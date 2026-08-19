import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/no-click-by-text.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module" },
});

ruleTester.run("no-click-by-text", rule as never, {
  valid: [
    'await user.click(screen.getByRole("option", { name: "Block SSN" }));',
    'await user.click(await screen.findByRole("button", { name: /save/i }));',
    'await user.click(screen.getByLabelText("Template"));',
    'expect(screen.getByText("Block SSN")).toBeInTheDocument();',
    'const node = screen.getByText("Block SSN");',
    'await user.type(screen.getByPlaceholderText("url"), "x");',
  ],
  invalid: [
    {
      code: 'await user.click(screen.getByText("Block SSN"));',
      errors: [{ messageId: "clickByText", data: { query: "getByText" } }],
    },
    {
      code: 'await user.click(await screen.findByText("Yes"));',
      errors: [{ messageId: "clickByText", data: { query: "findByText" } }],
    },
    {
      code: 'fireEvent.click(screen.getByText("Save"));',
      errors: [{ messageId: "clickByText", data: { query: "getByText" } }],
    },
    {
      code: 'await user.hover(screen.getAllByText("Row")[0]);',
      errors: [{ messageId: "clickByText", data: { query: "getAllByText" } }],
    },
  ],
});
