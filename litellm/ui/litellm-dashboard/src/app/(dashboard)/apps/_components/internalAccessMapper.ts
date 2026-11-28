import type { GranteeKind } from "./appAccessApi";

export const initialInternalAccessValues = { draft: null };

interface StoredAccess {
  object_permission?: { apps?: string[] | null } | null;
}

export const grantedAppIds = (stored: StoredAccess | null | undefined): string[] =>
  stored?.object_permission?.apps ?? [];

export const granteeAppsWrite = (kind: GranteeKind, id: string, apps: string[]) => {
  const object_permission = { apps };
  switch (kind) {
    case "team":
      return { kind, body: { team_id: id, object_permission } };
    case "user":
      return { kind, body: { user_id: id, object_permission } };
    case "organization":
      return { kind, body: { organization_id: id, object_permission } };
  }
};
