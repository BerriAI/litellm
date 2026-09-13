"use client";

import { useState } from "react";

import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useProjects } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import type { components } from "@/lib/http/schema";

type Target = components["schemas"]["MemoryPolicyInput"]["target_type"];
type PickerProps = Readonly<{ value: string; onChange: (value: string) => void; disabled: boolean }>;

function TeamPicker({ value, onChange, disabled }: PickerProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteTeams(25, search);
  return (
    <PaginatedSearchSelect
      inputId="memory-target"
      value={value}
      onValueChange={(v) => onChange(v ?? "")}
      options={(query.data?.pages ?? []).flatMap((page) =>
        page.teams.map((team) => ({ value: team.team_id, label: team.team_alias || team.team_id })),
      )}
      onSearchChange={setSearch}
      onLoadMore={query.fetchNextPage}
      hasNextPage={query.hasNextPage}
      isLoading={query.isLoading}
      isFetchingNextPage={query.isFetchingNextPage}
      disabled={disabled}
      errorText={query.error?.message}
      placeholder="Search teams"
    />
  );
}

function UserPicker({ value, onChange, disabled }: PickerProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteUsers(25, search);
  return (
    <PaginatedSearchSelect
      inputId="memory-target"
      value={value}
      onValueChange={(v) => onChange(v ?? "")}
      options={(query.data?.pages ?? []).flatMap((page) =>
        page.users.map((user) => ({ value: user.user_id, label: user.user_email || user.user_id })),
      )}
      onSearchChange={setSearch}
      onLoadMore={query.fetchNextPage}
      hasNextPage={query.hasNextPage}
      isLoading={query.isLoading}
      isFetchingNextPage={query.isFetchingNextPage}
      disabled={disabled}
      errorText={query.error?.message}
      placeholder="Search users by email"
    />
  );
}

export function MemoryKeyPicker({
  value,
  onChange,
  disabled,
  inputId = "memory-target",
  userId,
}: PickerProps & Readonly<{ inputId?: string; userId?: string }>) {
  const [search, setSearch] = useState("");
  const keyOptions = { search, userID: userId, includeTeamKeys: !userId, includeCreatedByKeys: !userId };
  const query = useInfiniteKeys(25, keyOptions);
  return (
    <PaginatedSearchSelect
      inputId={inputId}
      value={value}
      onValueChange={(v) => onChange(v ?? "")}
      options={(query.data?.pages ?? []).flatMap((page) =>
        page.keys.map((key) => ({ value: key.token, label: key.key_alias || key.key_name || key.token })),
      )}
      onSearchChange={setSearch}
      onLoadMore={query.fetchNextPage}
      hasNextPage={query.hasNextPage}
      isLoading={query.isLoading}
      isFetchingNextPage={query.isFetchingNextPage}
      disabled={disabled}
      errorText={query.error?.message}
      placeholder="Search virtual keys"
    />
  );
}

function OrganizationPicker({ value, onChange, disabled }: PickerProps) {
  const query = useOrganizations();
  return (
    <SearchSelect
      inputId="memory-target"
      value={value}
      onValueChange={(v) => onChange(v ?? "")}
      options={(query.data ?? []).map((org) => ({
        value: org.organization_id,
        label: org.organization_alias || org.organization_id,
      }))}
      disabled={disabled}
      placeholder="Select an organization"
      emptyText={query.error?.message ?? "No organizations found"}
    />
  );
}

function ProjectPicker({ value, onChange, disabled }: PickerProps) {
  const query = useProjects();
  return (
    <SearchSelect
      inputId="memory-target"
      value={value}
      onValueChange={(v) => onChange(v ?? "")}
      options={(query.data ?? []).map((project) => ({
        value: project.project_id,
        label: project.project_alias || project.project_id,
      }))}
      disabled={disabled}
      placeholder="Select a project"
      emptyText={query.error?.message ?? "No projects found"}
    />
  );
}

export function MemoryTargetPicker({ target, ...props }: PickerProps & Readonly<{ target: Target }>) {
  if (target === "team") return <TeamPicker {...props} />;
  if (target === "user") return <UserPicker {...props} />;
  if (target === "key") return <MemoryKeyPicker {...props} />;
  if (target === "organization") return <OrganizationPicker {...props} />;
  if (target === "project") return <ProjectPicker {...props} />;
  return null;
}
