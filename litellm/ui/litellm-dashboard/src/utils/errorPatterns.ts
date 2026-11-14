import type { ParseKeys } from "i18next";

export interface ErrorPattern {
  pattern: RegExp;
  replacement: ParseKeys;
}

// Centralised list of common error patterns.
// Add new patterns here without touching component files.
export const errorPatterns: ErrorPattern[] = [
  // Generic missing API key (covers OpenAI, Anthropic, etc.)
  {
    pattern: /Missing .* API Key/i,
    replacement: "errorPatterns.missingApiKey",
  },
  // Network / connectivity issues
  {
    pattern: /Connection timeout/i,
    replacement: "errorPatterns.connectionTimeout",
  },
  {
    pattern: /Network.*not.*ok/i,
    replacement: "errorPatterns.networkFailed",
  },
  // HTTP status based errors
  {
    pattern: /403.*Forbidden/i,
    replacement: "errorPatterns.forbidden",
  },
  {
    pattern: /401.*Unauthorized/i,
    replacement: "errorPatterns.unauthorized",
  },
  {
    pattern: /429.*rate limit/i,
    replacement: "errorPatterns.rateLimitExceeded",
  },
  {
    pattern: /500.*Internal Server Error/i,
    replacement: "errorPatterns.providerServerError",
  },
  // LiteLLM specific wrapped errors
  {
    pattern: /litellm\.AuthenticationError/i,
    replacement: "errorPatterns.authenticationFailed",
  },
  {
    pattern: /litellm\.RateLimitError/i,
    replacement: "errorPatterns.rateLimitExceeded",
  },
  {
    pattern: /litellm\.APIError/i,
    replacement: "errorPatterns.apiError",
  },
];
