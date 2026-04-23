/**
 * Local ESLint plugin for the phase-1 shadcn UI migration.
 *
 * Exposes two rules:
 *   - `litellm-ui/no-banned-ui-imports`
 *   - `litellm-ui/no-raw-tailwind-colors`
 *
 * Registered via a file: dep in package.json:
 *   "eslint-plugin-litellm-ui": "file:./eslint-plugin-litellm-ui"
 */
"use strict";

module.exports = {
  rules: {
    "no-banned-ui-imports": require("./rules/no-banned-ui-imports"),
    "no-raw-tailwind-colors": require("./rules/no-raw-tailwind-colors"),
  },
};
