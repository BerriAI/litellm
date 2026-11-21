"use client";

import { useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import {
  WorkspaceSwitchers,
  useWorkspaceScope,
  type WorkspaceItem,
  type WorkspacePicker,
  type WorkspaceKind,
} from "@litellm/lens-ui";
import { useOrganizations, useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useAllTeams, useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useProjects, type ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useProjectDetails } from "@/app/(dashboard)/hooks/projects/useProjectDetails";
import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeyInfo";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import type { Organization } from "@/components/networking";

const isKeyHash = (value: string) => /^[a-f0-9]{64}$/i.test(value);
const unresolved = (id: string | null | undefined, items: readonly WorkspaceItem[]) =>
  id ? items.find((item) => item.id === id) ?? { id, name: "Selected workspace" } : null;

function organizationItem(organization: Organization): WorkspaceItem {
  return { id: organization.organization_id, name: organization.organization_alias || "Unnamed organization" };
}

function teamItem(team: Team): WorkspaceItem {
  return {
    id: team.team_id,
    name: team.team_alias || "Unnamed team",
    organization_id: team.organization_id || null,
  };
}

function teamOrganization(teams: readonly Team[], teamId: string | null): string | null {
  return teams.find((team) => team.team_id === teamId)?.organization_id || null;
}

function projectItem(project: ProjectResponse, teams: readonly Team[]): WorkspaceItem {
  return {
    id: project.project_id,
    name: project.project_alias || "Unnamed project",
    team_id: project.team_id,
    organization_id: teamOrganization(teams, project.team_id),
  };
}

function inheritedTeam(key: KeyResponse | undefined, project: ProjectResponse | undefined) {
  return key?.team_id ?? project?.team_id ?? undefined;
}

function inheritedOrganization(key: KeyResponse | undefined, team: Team | undefined) {
  return key?.organization_id ?? key?.org_id ?? team?.organization_id ?? undefined;
}

function keyItem(key: KeyResponse, teams: readonly Team[]): WorkspaceItem {
  return {
    id: key.token,
    name: key.key_alias || key.key_name || "Unnamed key",
    organization_id: key.organization_id || key.org_id || teamOrganization(teams, key.team_id),
    team_id: key.team_id,
    project_id: key.project_id ?? null,
  };
}

function pickerQueryState(
  kind: WorkspaceKind,
  list: Pick<UseQueryResult, "isFetching" | "isError" | "refetch">,
  selected: Pick<UseQueryResult, "isFetching">,
): Pick<WorkspacePicker, "loading" | "error" | "onRetry"> {
  return {
    loading: list.isFetching || selected.isFetching,
    error: list.isError ? `Could not load ${kind}s.` : null,
    onRetry: () => void list.refetch(),
  };
}

export function GatewayWorkspaceSwitcher() {
  const { scope, select } = useWorkspaceScope();
  const [search, setSearch] = useState("");
  const [keySearch] = useDebouncedValue(search, { wait: 250 });
  const organizations = useOrganizations();
  const teams = useAllTeams();
  const projects = useProjects();
  const selectedKey = useKeyInfo(scope.api_key_hash ?? null, {
    enabled: Boolean(scope.api_key_hash && isKeyHash(scope.api_key_hash)),
  });
  const projectId = scope.project_id ?? selectedKey.data?.project_id ?? undefined;
  const selectedProject = useProjectDetails(projectId);
  const teamId = scope.team_id ?? inheritedTeam(selectedKey.data, selectedProject.data);
  const selectedTeam = useTeam(teamId);
  const organizationId = scope.organization_id ?? inheritedOrganization(selectedKey.data, selectedTeam.data);
  const selectedOrganization = useOrganization(organizationId);
  const allTeams = teams.data ?? [];
  const organizationItems = (organizations.data ?? []).map(organizationItem);
  const teamItems: WorkspaceItem[] = allTeams
    .filter((team) => !organizationId || team.organization_id === organizationId)
    .map(teamItem);
  const projectItems: WorkspaceItem[] = (projects.data ?? [])
    .filter((project) => {
      if (teamId) return project.team_id === teamId;
      if (organizationId)
        return allTeams.some((team) => team.team_id === project.team_id && team.organization_id === organizationId);
      return true;
    })
    .map((project) => projectItem(project, allTeams));
  const keyListOptions = {
    organizationID: organizationId,
    teamID: teamId,
    projectID: projectId,
    search: keySearch || undefined,
  };
  const keys = useInfiniteKeys(50, keyListOptions);
  const keyItems = (keys.data?.pages.flatMap((page) => page.keys) ?? [])
    .filter((key) => isKeyHash(key.token))
    .map((key) => keyItem(key, allTeams));
  const organizationValue = selectedOrganization.data
    ? organizationItem(selectedOrganization.data)
    : unresolved(organizationId, organizationItems);
  const teamValue = selectedTeam.data ? teamItem(selectedTeam.data) : unresolved(teamId, teamItems);
  const projectValue = selectedProject.data
    ? projectItem(selectedProject.data, allTeams)
    : unresolved(projectId, projectItems);
  const keyValue = selectedKey.data ? keyItem(selectedKey.data, allTeams) : unresolved(scope.api_key_hash, keyItems);
  const pickers: WorkspacePicker[] = [
    {
      kind: "organization",
      value: organizationValue,
      items: organizationItems,
      ...pickerQueryState("organization", organizations, selectedOrganization),
    },
    {
      kind: "team",
      value: teamValue,
      items: teamItems,
      ...pickerQueryState("team", teams, selectedTeam),
    },
    {
      kind: "project",
      value: projectValue,
      items: projectItems,
      ...pickerQueryState("project", projects, selectedProject),
    },
    {
      kind: "key",
      value: keyValue,
      items: keyItems,
      ...pickerQueryState("key", keys, selectedKey),
      query: search,
      onSearch: setSearch,
      hasMore: keys.hasNextPage,
      onLoadMore: () => void keys.fetchNextPage(),
    },
  ];
  const effectiveScope = { ...scope, organization_id: organizationId, team_id: teamId, project_id: projectId };
  return <WorkspaceSwitchers pickers={pickers} onSelect={(kind, item) => select(kind, item, effectiveScope)} />;
}
