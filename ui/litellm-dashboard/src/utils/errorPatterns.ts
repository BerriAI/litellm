export interface ErrorPattern {
  pattern: RegExp;
  replacement: string;
}

// Centralised list of common error patterns.
// Add new patterns here without touching component files.
export const errorPatterns: ErrorPattern[] = [
  // Generic missing API key (covers OpenAI, Anthropic, etc.)
  {
    pattern: /Missing .* API Key/i,
    replacement: "Missing API Key",
  },
  // Network / connectivity issues
  {
    pattern: /Connection timeout/i,
    replacement: "Connection timeout",
  },
  {
    pattern: /Network.*not.*ok/i,
    replacement: "Network connection failed",
  },
  // HTTP status based errors
  {
    pattern: /403.*Forbidden/i,
    replacement: "Access forbidden - check API key permissions",
  },
  {
    pattern: /401.*Unauthorized/i,
    replacement: "Unauthorized - invalid API key",
  },
  {
    pattern: /429.*rate limit/i,
    replacement: "Rate limit exceeded",
  },
  {
    pattern: /500.*Internal Server Error/i,
    replacement: "Provider internal server error",
  },
  // LiteLLM specific wrapped errors
  {
    pattern: /litellm\.AuthenticationError/i,
    replacement: "Authentication failed",
  },
  {
    pattern: /litellm\.RateLimitError/i,
    replacement: "Rate limit exceeded",
  },
  {
    pattern: /litellm\.APIError/i,
    replacement: "API error",
  },
];
