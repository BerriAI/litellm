import js from "@eslint/js";
import tseslint from "typescript-eslint";
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import prettier from "eslint-config-prettier/flat";
import unusedImports from "eslint-plugin-unused-imports";
import testingLibrary from "eslint-plugin-testing-library";
import jestDom from "eslint-plugin-jest-dom";
import local from "./scripts/eslint-rules/index.mjs";

const eslintConfig = [
  {
    ignores: [".next/**", "out/**", "build/**", "coverage/**", "next-env.d.ts", "src/lib/http/schema.d.ts"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...nextCoreWebVitals,
  prettier,
  {
    plugins: { "unused-imports": unusedImports, local },
    rules: {
      "unused-imports/no-unused-imports": "error",
      "local/no-large-inline-object-arg": "warn",
      "local/no-long-condition-chain": "warn",
      "local/no-complex-jsx-arrow": ["error", { maxStatements: 2 }],
      "local/no-noop-hover-variant": "error",
      "@typescript-eslint/no-explicit-any": "warn",
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-unused-expressions": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "prefer-const": "error",
      "no-empty": "off",
      "no-prototype-builtins": "off",
      "no-useless-catch": "off",
      "no-useless-escape": "off",
      "no-self-assign": "error",
      "no-var": "error",
      "no-nested-ternary": "error",
      "react/no-danger": "error",
      complexity: ["warn", 20],
      "max-depth": ["warn", 4],
      "max-params": ["error", 4],
      "max-nested-callbacks": ["error", 4],
      "no-restricted-syntax": [
        "error",
        {
          selector: "CallExpression[callee.name='fetch']",
          message:
            "Raw fetch() is only allowed in src/lib/http/. Use the shared client (createApiClient / apiClient) from @/lib/http/client instead.",
        },
      ],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@tremor/react", "@tremor/react/*"],
              message:
                "@tremor/react is being phased out; build new UI with shadcn/ui primitives instead of adding tremor imports.",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/**/*.tsx"],
    rules: {
      "local/filename-pascal-case": "error",
    },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    ignores: ["src/**/*.test.{ts,tsx}", "src/**/*.spec.{ts,tsx}", "src/data/**"],
    rules: {
      "max-lines": ["error", { max: 800, skipBlankLines: true, skipComments: true }],
    },
  },
  {
    files: ["src/lib/http/**"],
    rules: {
      "no-restricted-syntax": "off",
    },
  },
  {
    files: ["src/**/*.{ts,tsx}", "tests/**/*.{ts,tsx}"],
    rules: { "local/no-ad-hoc-z-index": "error" },
  },
  {
    files: [
      "src/components/ui/**/*.{ts,tsx}",
      "src/components/shared/DataTable/**/*.{ts,tsx}",
      "src/**/*.test.{ts,tsx}",
      "tests/**/*.{ts,tsx}",
    ],
    rules: { "local/no-ad-hoc-z-index": ["error", { allowPopupLayer: true }] },
  },
  {
    files: ["tests/eslint-rules/**/*.{ts,tsx}"],
    rules: { "local/no-noop-hover-variant": "off", "local/no-ad-hoc-z-index": "off" },
  },
  {
    files: ["src/**/*.test.{ts,tsx}", "tests/**/*.{ts,tsx}"],
    plugins: { "testing-library": testingLibrary, "jest-dom": jestDom },
    rules: {
      "testing-library/await-async-queries": "error",
      "testing-library/no-wait-for-multiple-assertions": "error",
      "testing-library/no-wait-for-side-effects": "error",
      "testing-library/prefer-find-by": "error",
      "testing-library/prefer-presence-queries": "error",
      "jest-dom/prefer-checked": "error",
      "jest-dom/prefer-empty": "error",
      "jest-dom/prefer-enabled-disabled": "error",
      "jest-dom/prefer-focus": "error",
      "jest-dom/prefer-in-document": "error",
      "jest-dom/prefer-to-have-attribute": "error",
      "jest-dom/prefer-to-have-class": "error",
      "jest-dom/prefer-to-have-style": "error",
      "jest-dom/prefer-to-have-text-content": "error",
    },
  },
];

export default eslintConfig;
