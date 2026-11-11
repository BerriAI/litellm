import js from "@eslint/js";
import tseslint from "typescript-eslint";
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import prettier from "eslint-config-prettier/flat";
import unusedImports from "eslint-plugin-unused-imports";
import testingLibrary from "eslint-plugin-testing-library";
import jestDom from "eslint-plugin-jest-dom";
import i18next from "eslint-plugin-i18next";
import htmlEntities from "eslint-plugin-i18next/lib/options/htmlEntities.js";
import local from "./scripts/eslint-rules/index.mjs";

const rawFetchRestriction = {
  selector: "CallExpression[callee.name='fetch']",
  message:
    "Raw fetch() is only allowed in src/lib/http/. Use the shared client (createApiClient / apiClient) from @/lib/http/client instead.",
};

const toastLiteralRestriction = {
  selector:
    "CallExpression[callee.object.name='toast'][callee.property.name=/^(success|error|warning|info)$/] > Literal.arguments:first-child",
  message: "Pass toast a t() string, not an English literal, so the message can be translated.",
};

const noLiteralStringOptions = {
  mode: "jsx-only",
  "should-validate-template": true,
  "jsx-components": { exclude: ["Trans", "code", "pre", "kbd", "Kbd", "CodeBlock", "SyntaxHighlighter"] },
  "jsx-attributes": {
    exclude: [
      "className",
      ".*ClassName",
      "style",
      ".*Style",
      "key",
      "id",
      ".*Id",
      ".*ID",
      "htmlFor",
      "name",
      "value",
      "defaultValue",
      "data-.*",
      "aria-(describedby|labelledby|controls|owns|haspopup|live|current|hidden|expanded|selected)",
      "type",
      "role",
      "href",
      "to",
      "src",
      "target",
      "rel",
      "accept",
      "autoComplete",
      "inputMode",
      "step",
      "min",
      "max",
      "width",
      "height",
      "size",
      "color",
      ".*Color",
      "variant",
      "language",
      "index",
      ".*Key",
      "keys",
      "feature",
      "fill",
      "stroke",
      "scope",
      "delay",
      "stackId",
      "category",
      "categories",
      "gridCols",
      "defaultTheme",
      "defaultRole",
      "schemaComponent",
      "storageKey",
      "code",
    ],
  },
  callees: {
    exclude: [
      "i18n(ext)?",
      "t",
      "i18n\\.t",
      "require",
      "addEventListener",
      "removeEventListener",
      "postMessage",
      "getElementById",
      "dispatch",
      "commit",
      "includes",
      "indexOf",
      "endsWith",
      "startsWith",
      "cn",
      "clsx",
      "cva",
      ".*\\.(get|set|has|delete|getAll|append|getColumn|format|split|replace|closest|querySelector|getItem|setItem|removeItem|toLocaleDateString|toLocaleString|hasOwnProperty)",
      "Object\\.hasOwn",
      "uiHref",
      "hasVisited",
      "toggleKey",
      "emitLocalStorageChange",
      "setLocalStorageItem",
      "removeLocalStorageItem",
      "getLocalStorageItem",
      "window\\.open",
      "Error",
      "new Error",
    ],
  },
  words: {
    exclude: [
      "[0-9!-/:-@[-`{-~]+",
      "[A-Z_-]+",
      "v",
      htmlEntities,
      /^\p{Emoji}+$/u,
      /^[\p{P}\p{S}\p{N}\p{M}\s]+$/u,
      /^[a-z0-9]+(?:[_\-.:/][a-z0-9*[\]]+)+:?$/,
      /^(?:https?:\/\/|\/|~\/|\.\/)\S*$/,
    ],
  },
  "object-properties": {
    exclude: [
      "[A-Z_-]+",
      "value",
      "key",
      "id",
      "name",
      "dataIndex",
      "className",
      "width",
      "fontSize",
      "maxHeight",
      "borderRadius",
      "bg",
      "color",
      "background",
      "padding",
      "fill",
      "role",
      "class",
      "spellcheck",
      "toast",
      "scope",
      "origin",
      "classifierType",
      "defaultValue",
      "1",
    ],
  },
};

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
      "no-restricted-syntax": ["error", rawFetchRestriction],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@tremor/react", "@tremor/react/*"],
              message:
                "@tremor/react is being phased out; build new UI with shadcn/ui primitives instead of adding tremor imports.",
            },
            {
              group: ["zod/*"],
              message: 'Import Zod from "zod"; the dashboard uses Zod 4 only.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/**/*.tsx"],
    ignores: ["src/components/ui/**"],
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
    files: ["src/**/*.{ts,tsx}"],
    ignores: [
      "src/**/*.test.{ts,tsx}",
      "src/**/*.spec.{ts,tsx}",
      "src/**/*.test-d.ts",
      "src/**/__tests__/**",
      "src/**/*TestHarness.tsx",
    ],
    languageOptions: { parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname } },
    plugins: { i18next },
    rules: {
      "i18next/no-literal-string": ["error", noLiteralStringOptions],
      "no-restricted-syntax": ["error", rawFetchRestriction, toastLiteralRestriction],
    },
  },
  {
    files: ["src/i18n/untranslated/**"],
    rules: { "i18next/no-literal-string": "off" },
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
    files: ["src/components/lens/**/*.tsx"],
    ignores: ["src/**/*.test.tsx"],
    rules: { "local/no-arbitrary-design-value": "error" },
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
      "testing-library/no-container": "warn",
      "testing-library/no-node-access": "warn",
      "testing-library/no-wait-for-multiple-assertions": "error",
      "testing-library/no-wait-for-side-effects": "error",
      "testing-library/prefer-find-by": "error",
      "testing-library/prefer-presence-queries": "error",
      "testing-library/prefer-screen-queries": "warn",
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
