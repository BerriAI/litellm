import React, { useMemo, useState } from "react";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../key_team_helpers/key_list";

interface TeamDropdownProps {
  value?: string;
  onChange?: (value: string) => void;
  /** Callback with the full Team object (or null on clear). */
  onTeamSelect?: (team: Team | null) => void;
  disabled?: boolean;
  /** Filter teams by organization. */
  organizationId?: string | null;
  pageSize?: number;
  id?: string;
}

const TeamDropdown: React.FC<TeamDropdownProps> = ({
  value,
  onChange,
  onTeamSelect,
  disabled,
  organizationId,
  pageSize = 20,
  id,
}) => {
  const [search, setSearch] = useState("");

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteTeams(
    pageSize,
    search || undefined,
    organizationId,
  );

  const teams = useMemo(() => {
    if (!data?.pages) return [];
    const seen = new Set<string>();
    const result: Team[] = [];
    for (const page of data.pages) {
      for (const team of page.teams) {
        if (seen.has(team.team_id)) continue;
        seen.add(team.team_id);
        result.push(team);
      }
    }
    return result;
  }, [data]);

  const handleChange = (teamId: string) => {
    onChange?.(teamId);
    if (onTeamSelect) {
      onTeamSelect(teamId ? teams.find((t) => t.team_id === teamId) ?? null : null);
    }
  };

  return (
    <div data-testid="team-dropdown">
      <PaginatedSearchSelect
        options={teams.map((team) => ({
          label: team.team_alias || team.team_id,
          value: team.team_id,
          sublabel: team.team_id,
        }))}
        value={value || undefined}
        onValueChange={handleChange}
        onSearchChange={setSearch}
        onLoadMore={fetchNextPage}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder="Search or select a team"
        emptyText="No teams found"
        loadingText="Loading teams…"
        disabled={disabled}
        inputId={id}
      />
    </div>
  );
};

export default TeamDropdown;
