/**
 * Key management types — generated from the backend OpenAPI schema.
 *
 * To regenerate: `npm run generate:all` from ui/litellm-dashboard/
 */
import type { components, paths } from "./api.generated";

// -- /key/block --
export type BlockKeyRequest = components["schemas"]["BlockKeyRequest"];
export type BlockKeyResponse =
  paths["/key/block"]["post"]["responses"]["200"]["content"]["application/json"];

// -- /key/unblock --
export type UnblockKeyRequest = components["schemas"]["BlockKeyRequest"]; // same request model
export type UnblockKeyResponse =
  paths["/key/unblock"]["post"]["responses"]["200"]["content"]["application/json"];
