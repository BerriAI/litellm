import React, { useEffect, useMemo, useState } from "react";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../key_team_helpers/key_list";

interface TeamDropdownProps {
  value?: string | null;
  onChange?: (value: string | null) => void;
  /** Callback with the full Team object (or null on clear). */
  onTeamSelect?: (team: Team | null) => void;
  disabled?: boolean;
  /** Filter teams by organization. */
  organizationId?: string | null;
  pageSize?: number;
  id?: string;
  filterTeam?: (team: Team) => boolean;
}

const TeamDropdown: React.FC<TeamDropdownProps> = ({
  value,
  onChange,
  onTeamSelect,
  disabled,
  organizationId,
  pageSize = 20,
  id,
  filterTeam,
}) => {
  const [search, setSearch] = useState("");

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isFetchNextPageError, isLoading } = useInfiniteTeams(
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

  const eligibleTeams = useMemo(() => teams.filter((team) => !filterTeam || filterTeam(team)), [teams, filterTeam]);
  const hasTeamFilter = filterTeam != null;

  useEffect(() => {
    if (
      hasTeamFilter &&
      eligibleTeams.length < pageSize &&
      hasNextPage &&
      !isLoading &&
      !isFetchingNextPage &&
      !isFetchNextPageError
    ) {
      void fetchNextPage();
    }
  }, [
    hasTeamFilter,
    eligibleTeams.length,
    pageSize,
    hasNextPage,
    isLoading,
    isFetchingNextPage,
    isFetchNextPageError,
    fetchNextPage,
  ]);

  const handleChange = (teamId: string | null) => {
    onChange?.(teamId);
    if (onTeamSelect) {
      onTeamSelect(teamId ? teams.find((t) => t.team_id === teamId) ?? null : null);
    }
  };

  return (
    <div data-testid="team-dropdown">
      <PaginatedSearchSelect
        options={eligibleTeams.map((team) => ({
          label: team.team_alias || team.team_id,
          value: team.team_id,
          sublabel: team.team_id,
        }))}
        value={value}
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
