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
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-unused-expressions": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "prefer-const": "off",
      "no-empty": "off",
      "no-prototype-builtins": "off",
      "no-useless-catch": "off",
      "no-useless-escape": "off",
      "no-self-assign": "off",
    },
  },
];

export default eslintConfig;
