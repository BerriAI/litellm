/**
 * Key management types — derived from the backend OpenAPI schema.
 *
 * Depends on api.generated.ts which is gitignored.
 * Run `npm run generate:types` from ui/litellm-dashboard/ to generate it.
 */
import type { components, paths } from "./api.generated";

// -- /key/block --
export type BlockKeyRequest = components["schemas"]["BlockKeyRequest"];
export type BlockKeyResponse =
  paths["/key/block"]["post"]["responses"]["200"]["content"]["application/json"];

// -- /key/unblock --
// Note: unblock_key is missing a return type annotation on the backend,
// so UnblockKeyResponse resolves to `unknown`. Fix: add
// `-> Optional[LiteLLM_VerificationToken]` to the unblock_key endpoint.
export type UnblockKeyRequest = components["schemas"]["BlockKeyRequest"]; // same request model
