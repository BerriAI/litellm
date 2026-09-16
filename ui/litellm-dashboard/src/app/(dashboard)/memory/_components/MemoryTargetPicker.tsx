"use client";

import { useState } from "react";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";

type PickerProps = Readonly<{
  inputId?: string;
  value: string;
  onChange: (value: string, label?: string) => void;
  disabled: boolean;
}>;

export function MemoryTeamPicker({ value, onChange, disabled, inputId = "memory-team" }: PickerProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteTeams(25, search);
  const options = (query.data?.pages ?? []).flatMap((page) =>
    page.teams.map((team) => ({ value: team.team_id, label: team.team_alias || team.team_id })),
  );
  return (
    <PaginatedSearchSelect
      inputId={inputId}
      value={value}
      onValueChange={(v) => onChange(v ?? "", options.find((option) => option.value === v)?.label)}
      options={options}
      onSearchChange={setSearch}
      onLoadMore={query.fetchNextPage}
      hasNextPage={query.hasNextPage}
      isLoading={query.isLoading}
      isFetchingNextPage={query.isFetchingNextPage}
      disabled={disabled}
      errorText={query.error?.message}
      placeholder="All permitted teams"
    />
  );
}

export function MemoryUserPicker({ value, onChange, disabled, inputId = "memory-user" }: PickerProps) {
  const [search, setSearch] = useState("");
  const query = useInfiniteUsers(25, search);
  const options = (query.data?.pages ?? []).flatMap((page) =>
    page.users.map((user) => ({ value: user.user_id, label: user.user_email || user.user_id })),
  );
  return (
    <PaginatedSearchSelect
      inputId={inputId}
      value={value}
      onValueChange={(v) => onChange(v ?? "", options.find((option) => option.value === v)?.label)}
      options={options}
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
