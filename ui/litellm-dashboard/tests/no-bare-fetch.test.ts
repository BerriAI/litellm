import { RuleTester } from "eslint";
import { describe, it } from "vitest";
// @ts-expect-error - plain JS ESLint rule module without type declarations
import rule from "../eslint-rules/no-bare-fetch.mjs";

RuleTester.describe = describe;
RuleTester.it = it;
RuleTester.itOnly = it.only;

const ruleTester = new RuleTester({
  languageOptions: { ecmaVersion: 2022, sourceType: "module" },
});

ruleTester.run("no-bare-fetch", rule, {
  valid: [
    `useQuery({ queryFn: async () => { const r = await fetch("/x"); return r.json(); } });`,
    `useMutation({ mutationFn: (body) => fetch("/x", { method: "POST", body }) });`,
    `useQuery({ "queryFn": () => fetch("/x") });`,
    `client.fetch("/x");`,
  ],
  invalid: [
    { code: `function Component() { fetch("/x"); }`, errors: [{ messageId: "bareFetch" }] },
    { code: `export const getThing = () => fetch("/x");`, errors: [{ messageId: "bareFetch" }] },
    { code: `useQuery({ queryKey: [fetch("/x")] });`, errors: [{ messageId: "bareFetch" }] },
    { code: `useQuery({ ["queryFn"]: () => fetch("/x") });`, errors: [{ messageId: "bareFetch" }] },
  ],
});
