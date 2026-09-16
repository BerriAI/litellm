import type { Organization, Team } from "@/components/networking";
import { isOrgAdminForAnyOrg, isProxyAdminRole } from "@/utils/roles";
import { useId } from "react";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";

type KeyProjectFieldProps = {
  projectId: string | null | undefined;
  canDetach: boolean;
  pending: boolean;
  disabled: boolean;
  onToggle: () => void;
};

export function KeyProjectField({ projectId, canDetach, pending, disabled, onToggle }: KeyProjectFieldProps) {
  const id = useId();
  const { data: projects } = useProjects();
  const alias = projects?.find((project) => project.project_id === projectId)?.project_alias;
  const display = alias ? `${alias} (${projectId})` : projectId;
  return (
    <Field>
      <FieldLabel htmlFor={id}>Project</FieldLabel>
      <Input id={id} value={display ?? ""} disabled readOnly />
      {canDetach && (
        <>
          {pending && (
            <p className="text-sm text-muted-foreground">
              The project will be removed when you save. Team, organization, and key limits will stay the same.
            </p>
          )}
          <Button type="button" variant="outline" disabled={disabled} onClick={onToggle}>
            {pending ? "Keep project" : "Detach from project"}
          </Button>
        </>
      )}
    </Field>
  );
}

type ProjectKeyTeam = Pick<Team, "organization_id" | "members_with_roles"> & {
  team_member_permissions?: string[] | null;
};

export function canDetachKeyProject(
  team: ProjectKeyTeam | undefined,
  organizations: Organization[] | undefined,
  userID: string | null,
  userRole: string | null,
): boolean {
  if (isProxyAdminRole(userRole ?? "")) return true;
  const member = team?.members_with_roles?.find((candidate) => candidate.user_id === userID);
  if (member?.role === "admin") return true;
  const canUpdateKey = member != null && team?.team_member_permissions?.includes("/key/update");
  const keyOrganizations = organizations?.filter((org) => org.organization_id === team?.organization_id);
  return Boolean(canUpdateKey && isOrgAdminForAnyOrg(keyOrganizations, userID));
}
