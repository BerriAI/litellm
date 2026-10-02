import { RuleTester } from "eslint";
import tseslint from "typescript-eslint";
import { describe, it } from "vitest";
import rule from "../../scripts/eslint-rules/no-query-is-loading.mjs";

RuleTester.describe = describe;
RuleTester.it = it;
RuleTester.itOnly = it.only;

const ruleTester = new RuleTester({
  languageOptions: {
    parser: tseslint.parser,
    parserOptions: {
      projectService: { allowDefaultProject: ["query-lint.ts"] },
      tsconfigRootDir: process.cwd(),
    },
  },
});

const prefix =
  'import type { UseQueryResult, UseInfiniteQueryResult, UseMutationResult } from "@tanstack/react-query";';
const query = `${prefix} declare const query: UseQueryResult<string[]>;`;
const filename = `${process.cwd()}/query-lint.ts`;

ruleTester.run("no-query-is-loading", rule, {
  valid: [
    { filename, code: `${query} query.isFetching; query.isPending; query.isEnabled;` },
    { filename, code: "const state = { isLoading: true }; state.isLoading;" },
    { filename, code: `${prefix} declare const mutation: UseMutationResult; mutation.isPending;` },
    { filename, code: "const { isLoading } = { isLoading: false };" },
  ],
  invalid: [
    { filename, code: `${query} query.isInitialLoading;`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} const loading = query.isFetching;`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} const { isFetching: loading } = query;`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} query.isLoading;`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} query["isLoading"];`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} const { isLoading: loading } = query;`, errors: [{ messageId: "readiness" }] },
    { filename, code: `${query} const { query: { isLoading } } = { query };`, errors: [{ messageId: "readiness" }] },
    {
      filename,
      code: `${query} function render({ isLoading }: UseQueryResult) {}`,
      errors: [{ messageId: "readiness" }],
    },
    { filename, code: `${query} const alias = query; alias.isLoading;`, errors: [{ messageId: "readiness" }] },
    {
      filename,
      code: `${prefix} declare const query: UseInfiniteQueryResult; query.isLoading;`,
      errors: [{ messageId: "readiness" }],
    },
    {
      filename,
      code: `${prefix} declare function useCustomQuery(): UseQueryResult; const { isLoading } = useCustomQuery();`,
      errors: [{ messageId: "readiness" }],
    },
  ],
});
