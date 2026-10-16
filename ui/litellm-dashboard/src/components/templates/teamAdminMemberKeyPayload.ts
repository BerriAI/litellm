import { Member } from "@/components/networking";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";

export const KEY_BUDGET_FIELDS = ["max_budget", "soft_budget", "budget_duration", "budget_limits"] as const;

export const isTeamAdminEditingMemberKey = (args: {
  userRole: string;
  userId: string;
  keyUserId: string | null | undefined;
  keyTeamId: string | null | undefined;
  teamMembers: Member[] | null | undefined;
}): boolean => {
  if (isProxyAdminRole(args.userRole)) return false;
  if (!args.keyTeamId) return false;
  if (args.keyUserId === args.userId) return false;
  return isUserTeamAdminForSingleTeam(args.teamMembers ?? null, args.userId);
};

export type TeamAdminMemberKeyPayload =
  | { kind: "ok"; payload: Record<string, unknown> }
  | { kind: "blocked"; fields: readonly string[] };

export const teamAdminMemberKeyPayload = (
  formValues: Record<string, unknown>,
  dirtyFields: readonly string[],
): TeamAdminMemberKeyPayload => {
  const disallowed = dirtyFields.filter(
    (field) => field !== "token" && field !== "key" && !(KEY_BUDGET_FIELDS as readonly string[]).includes(field),
  );
  if (disallowed.length > 0) {
    return { kind: "blocked", fields: disallowed };
  }
  const payload: Record<string, unknown> = { key: formValues.key };
  for (const field of KEY_BUDGET_FIELDS) {
    if (!dirtyFields.includes(field)) continue;
    if (formValues[field] === undefined) continue;
    if (field === "budget_duration" && formValues[field] === "") continue;
    payload[field] = formValues[field];
  }
  return { kind: "ok", payload };
};
