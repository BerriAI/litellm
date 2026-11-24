import type { components } from "./controlPlaneSchema";

type Schema = components["schemas"];
export type ControlPlanePrincipal = Schema["PrincipalResponse"];
export type ControlPlaneDataplane = Schema["DataplaneResponse"];
export type ControlPlaneDataplaneInput = Schema["DataplaneSpec"];
export type ControlPlaneRoleInput = Schema["RoleSpec"];

export const controlPlaneSessionKey = (prefix: string) => `litellm_control_plane_session:${prefix}`;

export async function discoverControlPlane() {
  const response = await fetch("/litellm/.well-known/litellm-ui-config", {
    headers: { Accept: "application/json" },
    credentials: "omit",
    redirect: "error",
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new ControlPlaneError(response.status, "UI discovery failed.");
  const data: unknown = await response.json();
  if (!isRecord(data) || data.mode !== "control_plane_api") return null;
  const rootPath = typeof data.server_root_path === "string" ? data.server_root_path.replace(/\/$/, "") : "";
  const routePrefix = typeof data.control_plane_api_prefix === "string" ? data.control_plane_api_prefix : "";
  const hasInvalidRootPath = rootPath && !safePrefix(rootPath);
  if (hasInvalidRootPath || !safePrefix(routePrefix)) {
    throw new Error("The control plane API prefix is not a same-origin path.");
  }
  return {
    prefix: `${rootPath}${routePrefix}`,
    rootPath,
    version: typeof data.litellm_version === "string" ? data.litellm_version : null,
    disabled: "admin_ui_disabled" in data && data.admin_ui_disabled === true,
  };
}

function safePrefix(prefix: string): boolean {
  const segments = prefix.split("/").slice(1);
  const pathSegment = /^(?:[a-zA-Z0-9._~!$&'()*+,;=:@-]|(?=[^\p{ASCII}])[\p{L}\p{M}\p{N}\p{P}\p{S}])+$/u;
  return (
    prefix.startsWith("/") &&
    segments.length > 0 &&
    segments.every((segment) => pathSegment.test(segment) && segment !== "." && segment !== "..")
  );
}

export function controlPlaneResourceSegment(value: string) {
  const isDotSegment = value === "." || value === "..";
  if (!value || isDotSegment || /[/\\%]/.test(value)) {
    throw new Error("Invalid control plane resource identifier.");
  }
  return encodeURIComponent(value);
}

const segment = controlPlaneResourceSegment;

export class ControlPlaneError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(`${status}: ${message}`);
  }
}

export const hasScope = (principal: ControlPlanePrincipal, scope: string) => principal.scopes.includes(scope);

export const hasDataplane = (principal: ControlPlanePrincipal, id: string) =>
  principal.dataplanes === null || principal.dataplanes.includes(id);

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

export function controlPlaneApi(prefix: string, accessToken?: string, onSessionExpired?: () => void) {
  if (!safePrefix(prefix)) throw new Error("The control plane API prefix is not a same-origin path.");

  async function request<T>(path: string, method = "GET", body?: object, query?: URLSearchParams): Promise<T> {
    const search = query?.toString();
    const options: RequestInit = {
      method,
      headers: {
        Accept: "application/json",
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    };
    const response = await fetch(`${prefix}${path}${search ? `?${search}` : ""}`, options);
    if (!response.ok) {
      // The passthrough marks dataplane replies, including dataplane credential errors.
      if (response.status === 401 && accessToken && !response.headers.has("X-LiteLLM-Dataplane")) {
        onSessionExpired?.();
      }
      const error: unknown = await response.json().catch(() => null);
      const detail = isRecord(error) && typeof error.detail === "string" ? error.detail : response.statusText;
      throw new ControlPlaneError(response.status, detail || "Request failed");
    }
    return response.status === 204 ? (undefined as T) : response.json();
  }

  return {
    login: (username: string, password: string) =>
      request<Schema["SessionTokenResponse"]>("/auth/login", "POST", { username, password }),
    me: () => request<ControlPlanePrincipal>("/auth/me"),
    dataplanes: () => request<Schema["DataplaneListResponse"]>("/dataplanes"),
    health: (id: string) => request<Schema["DataplaneHealthResponse"]>(`/dataplanes/${segment(id)}/health`),
    models: () => request<Schema["FleetModelsResponse"]>("/fleet/models"),
    teams: () => request<Schema["FleetTeamsResponse"]>("/fleet/teams"),
    usageSummary: (query?: URLSearchParams) =>
      request<Schema["UsageSummaryResponse"]>("/usage/summary", "GET", undefined, query),
    usageDaily: (query?: URLSearchParams) =>
      request<Schema["DailyUsageResponse"]>("/usage/daily", "GET", undefined, query),
    usageTop: (query?: URLSearchParams) => request<Schema["TopUsageResponse"]>("/usage/top", "GET", undefined, query),
    roles: () => request<Schema["RoleListResponse"]>("/roles"),
    audit: (query?: URLSearchParams) => request<Schema["AuditListResponse"]>("/audit", "GET", undefined, query),
    saveDataplane: (id: string, data: ControlPlaneDataplaneInput) =>
      request<Schema["DataplaneResponse"]>(`/dataplanes/${segment(id)}`, "PUT", data),
    deleteDataplane: (id: string) => request<void>(`/dataplanes/${segment(id)}`, "DELETE"),
    saveRole: (name: string, data: ControlPlaneRoleInput) =>
      request<Schema["RoleResponse"]>(`/roles/${segment(name)}`, "PUT", data),
    deleteRole: (name: string) => request<void>(`/roles/${segment(name)}`, "DELETE"),
  };
}

export type ControlPlaneApi = ReturnType<typeof controlPlaneApi>;
