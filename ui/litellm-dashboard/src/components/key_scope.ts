export interface KeyModelScope {
  hasModelAccess: boolean;
  label: string | null;
}

const MANAGEMENT_ROUTES_PRESET = "management_routes";
const INFO_ROUTES_PRESET = "info_routes";
const SCIM_ROUTE_PREFIX = "/scim";

const isScimRoute = (route: string): boolean => route.startsWith(SCIM_ROUTE_PREFIX);

const isOnlyPreset = (allowedRoutes: string[], preset: string): boolean =>
  allowedRoutes.length === 1 && allowedRoutes[0] === preset;

/**
 * Derive whether a key can reach any LLM inference route from its allowed_routes,
 * and a human label for the scope when it cannot.
 *
 * key_type is not persisted on the key (the proxy maps it to allowed_routes and
 * drops it), so a key scoped to management / read-only / SCIM routes stores an
 * empty models list and would otherwise render as "All Proxy Models" even
 * though it cannot call a single model. Only the recognized no-model-access
 * scopes are reclassified here; unknown or custom allowed_routes (and the
 * unrestricted "full access" case) keep the default model-list rendering.
 */
export const deriveKeyModelScope = (allowedRoutes: string[] | null | undefined): KeyModelScope => {
  if (!Array.isArray(allowedRoutes) || allowedRoutes.length === 0) {
    return { hasModelAccess: true, label: null };
  }

  if (allowedRoutes.every(isScimRoute)) {
    return { hasModelAccess: false, label: "SCIM" };
  }

  if (isOnlyPreset(allowedRoutes, MANAGEMENT_ROUTES_PRESET)) {
    return { hasModelAccess: false, label: "Management" };
  }

  if (isOnlyPreset(allowedRoutes, INFO_ROUTES_PRESET)) {
    return { hasModelAccess: false, label: "Read-only" };
  }

  return { hasModelAccess: true, label: null };
};
