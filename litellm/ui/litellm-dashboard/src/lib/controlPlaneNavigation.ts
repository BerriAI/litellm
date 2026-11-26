import type { ParseKeys, TFunction } from "i18next";
import { hasScope, type ControlPlanePrincipal } from "./http/controlPlane";
import { routeSegmentForPathname } from "@/utils/uiHref";

export function controlPlaneRoute(pathname: string) {
  const route = routeSegmentForPathname(pathname);
  return route === "control-plane" ? `control-plane/${pathname.split("/").filter(Boolean).at(-1)}` : route;
}

export const controlPlanePages: Record<string, { label: ParseKeys; description: ParseKeys; scope: string }> = {
  "fleet-models": {
    label: "controlPlane.fleetModels",
    description: "controlPlane.fleetModelsDescription",
    scope: "fleet:read",
  },
  "fleet-teams": {
    label: "controlPlane.fleetTeams",
    description: "controlPlane.fleetTeamsDescription",
    scope: "fleet:read",
  },
  "fleet-usage": {
    label: "controlPlane.fleetUsage",
    description: "controlPlane.fleetUsageDescription",
    scope: "usage:read",
  },
  dataplanes: {
    label: "controlPlane.dataplanes",
    description: "controlPlane.dataplanesDescription",
    scope: "dataplanes:read",
  },
  roles: { label: "controlPlane.roles", description: "controlPlane.rolesDescription", scope: "roles:write" },
  audit: {
    label: "controlPlane.auditLog",
    description: "controlPlane.latest100EventsVisibleToThisSession",
    scope: "audit:read",
  },
};

const proxiedPages = new Set([
  "api-keys",
  "teams",
  "users",
  "models-and-endpoints",
  "organizations",
  "budgets",
  "guardrails",
  "usage",
  "mcp-servers",
  "credentials",
]);

export function controlPlanePageReason(
  route: string,
  session: { selected: string | null; principal: ControlPlanePrincipal },
  t: TFunction,
) {
  if (!route || route === "home") return undefined;
  const native = Object.hasOwn(controlPlanePages, route) ? controlPlanePages[route] : undefined;
  if (!native && !proxiedPages.has(route)) return t("controlPlane.pageUnavailable");
  const scope = native?.scope ?? "proxy:read";
  if (!hasScope(session.principal, scope)) return t("controlPlane.requiresScope", { scope });
  if (!native && !session.selected) return t("controlPlane.selectADataplaneToOpenThisPage");
}
