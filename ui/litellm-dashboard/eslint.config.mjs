import js from "@eslint/js";
import tseslint from "typescript-eslint";
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import prettier from "eslint-config-prettier/flat";
import unusedImports from "eslint-plugin-unused-imports";

const eslintConfig = [
  {
    ignores: [".next/**", "out/**", "build/**", "coverage/**", "next-env.d.ts"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...nextCoreWebVitals,
  prettier,
  {
    plugins: { "unused-imports": unusedImports },
    rules: {
      "unused-imports/no-unused-imports": "error",
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-unused-expressions": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "prefer-const": "off",
      "no-empty": "off",
      "no-prototype-builtins": "off",
      "no-useless-catch": "off",
      "no-useless-escape": "off",
      "no-self-assign": "error",
      "no-var": "error",
      "react/no-danger": "error",
      complexity: ["warn", 20],
      "max-depth": ["warn", 4],
      "max-params": ["error", 4],
      "max-nested-callbacks": ["error", 4],
      "no-restricted-syntax": [
        "error",
        {
          selector: "CallExpression[callee.name='fetch']",
          message: "Use React Query (@tanstack/react-query) for data fetching instead of a raw fetch().",
        },
      ],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@tremor/react", "@tremor/react/*"],
              message: "@tremor/react is being phased out; build new UI with antd instead of adding tremor imports.",
            },
          ],
        },
      ],
    },
  },
];

export default eslintConfig;
