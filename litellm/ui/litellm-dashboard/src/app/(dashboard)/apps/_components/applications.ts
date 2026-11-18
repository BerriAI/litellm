import type { App, AppResource } from "./appForms";

export interface ApplicationRow {
  resource: AppResource;
  app: App | null;
}

export const applicationRows = (
  resources: readonly AppResource[],
  apps: readonly App[],
  search: string,
): ApplicationRow[] => {
  const byId = new Map(apps.map((app) => [app.app_id, app]));
  const needle = search.trim().toLocaleLowerCase();
  return resources
    .filter(
      (resource) =>
        needle === "" ||
        resource.alias.toLocaleLowerCase().includes(needle) ||
        resource.resource_id.toLocaleLowerCase().includes(needle),
    )
    .map((resource) => ({ resource, app: (resource.app_id && byId.get(resource.app_id)) || null }))
    .sort((left, right) => left.resource.alias.localeCompare(right.resource.alias));
};

export const applicationInitials = (label: string): string =>
  label
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part.charAt(0))
    .join("")
    .toLocaleUpperCase();
