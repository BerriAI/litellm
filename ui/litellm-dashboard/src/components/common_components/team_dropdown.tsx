import React, { useEffect, useMemo, useState, type UIEvent } from "react";
import { Select, Typography } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { Team } from "../key_team_helpers/key_list";

const { Text } = Typography;

interface TeamDropdownProps {
  value?: string;
  onChange?: (value: string) => void;
  /** Callback with the full Team object (or null on clear). */
  onTeamSelect?: (team: Team | null) => void;
  disabled?: boolean;
  /** Filter teams by organization. */
  organizationId?: string | null;
  pageSize?: number;
  /** Only list teams the current user administers, e.g. for team-scoped model creation. */
  adminOnly?: boolean;
}

const SCROLL_THRESHOLD = 0.8;

const TeamDropdown: React.FC<TeamDropdownProps> = ({
  value,
  onChange,
  onTeamSelect,
  disabled,
  organizationId,
  pageSize = 20,
  adminOnly = false,
}) => {
  const { userId } = useAuthorized();
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_WAIT_MS,
  });

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } = useInfiniteTeams(
    pageSize,
    debouncedSearch || undefined,
    organizationId,
  );

  const teams = useMemo(() => {
    if (!data?.pages) return [];
    const seen = new Set<string>();
    const result: Team[] = [];
    for (const page of data.pages) {
      for (const team of page.teams) {
        if (seen.has(team.team_id)) continue;
        if (adminOnly && !isUserTeamAdminForSingleTeam(team.members_with_roles, userId ?? "")) continue;
        seen.add(team.team_id);
        result.push(team);
      }
    }
    return result;
  }, [data, adminOnly, userId]);

  // adminOnly filters client-side, so a page of entirely non-admin teams renders an empty,
  // unscrollable popup that can never reach the next page through onPopupScroll. Keep paging
  // automatically until a match turns up or there is nothing left to fetch.
  useEffect(() => {
    const foundNothingYet = teams.length === 0 && hasNextPage;
    const idle = !isFetchingNextPage && !isLoading;
    if (adminOnly && foundNothingYet && idle) {
      fetchNextPage();
    }
  }, [adminOnly, teams.length, hasNextPage, isFetchingNextPage, isLoading, fetchNextPage]);

  const handlePopupScroll = (e: UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (scrollRatio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  };

  const handleSearch = (val: string) => {
    setSearchInput(val);
    setDebouncedSearch(val);
  };

  const handleChange = (teamId: string | undefined) => {
    onChange?.(teamId ?? "");
    if (onTeamSelect) {
      const team = teamId ? teams.find((t) => t.team_id === teamId) ?? null : null;
      onTeamSelect(team);
    }
  };

  // True while a match could still turn up on a later page, so "No teams found" is deferred
  // until every page has actually been checked instead of flashing between page fetches.
  const foundNothingYet = teams.length === 0 && hasNextPage;
  const stillSearchingForMatch = foundNothingYet && (isFetchingNextPage || adminOnly);

  return (
    <Select
      showSearch
      placeholder="Search or select a team"
      value={value || undefined}
      onChange={handleChange}
      disabled={disabled}
      allowClear
      filterOption={false}
      onSearch={handleSearch}
      searchValue={searchInput}
      onPopupScroll={handlePopupScroll}
      loading={isLoading}
      notFoundContent={isLoading || stillSearchingForMatch ? <LoadingOutlined spin /> : "No teams found"}
      data-testid="team-dropdown"
      popupRender={(menu) => (
        <>
          {menu}
          {isFetchingNextPage && (
            <div style={{ textAlign: "center", padding: 8 }}>
              <LoadingOutlined spin />
            </div>
          )}
        </>
      )}
    >
      {teams.map((team) => (
        <Select.Option key={team.team_id} value={team.team_id}>
          <span className="font-medium">{team.team_alias}</span> <Text type="secondary">({team.team_id})</Text>
        </Select.Option>
      ))}
    </Select>
  );
};

export default TeamDropdown;
