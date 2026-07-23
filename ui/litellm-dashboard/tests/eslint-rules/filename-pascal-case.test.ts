import { RuleTester } from "eslint";
import rule from "../../scripts/eslint-rules/filename-pascal-case.mjs";

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: "latest", sourceType: "module", parserOptions: { ecmaFeatures: { jsx: true } } },
});

ruleTester.run("filename-pascal-case", rule as never, {
  valid: [
    { code: "export const x = 1;", filename: "src/components/TeamInfo.tsx" },
    { code: "export const x = 1;", filename: "src/components/Button.tsx" },
    { code: "export const x = 1;", filename: "src/app/teams/page.tsx" },
    { code: "export const x = 1;", filename: "src/app/teams/layout.tsx" },
    { code: "export const x = 1;", filename: "src/app/teams/not-found.tsx" },
    { code: "export const x = 1;", filename: "src/components/TeamInfo.test.tsx" },
    { code: "export const x = 1;", filename: "src/components/view_users.test.tsx" },
    { code: "export const x = 1;", filename: "src/components/TeamInfo.spec.tsx" },
  ],
  invalid: [
    {
      code: "export const x = 1;",
      filename: "src/components/user_info_view.tsx",
      errors: [{ messageId: "notPascalCase", data: { name: "user_info_view.tsx", suggestion: "UserInfoView" } }],
    },
    {
      code: "export const x = 1;",
      filename: "src/components/team-info.tsx",
      errors: [{ messageId: "notPascalCase", data: { name: "team-info.tsx", suggestion: "TeamInfo" } }],
    },
    {
      code: "export const x = 1;",
      filename: "src/components/teamInfo.tsx",
      errors: [{ messageId: "notPascalCase", data: { name: "teamInfo.tsx", suggestion: "TeamInfo" } }],
    },
  ],
});
