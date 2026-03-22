import React, { useMemo, useState, type UIEvent } from "react";
import { Select } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
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
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

const TeamDropdown: React.FC<TeamDropdownProps> = ({
  value,
  onChange,
  onTeamSelect,
  disabled,
  organizationId,
  pageSize = 50,
}) => {
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useDebouncedState("", {
    wait: DEBOUNCE_MS,
  });

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
  } = useInfiniteTeams(
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
        seen.add(team.team_id);
        result.push(team);
      }
    }
    return result;
  }, [data]);

  const options = useMemo(
    () =>
      teams.map((team) => ({
        label: `${team.team_alias} (${team.team_id})`,
        value: team.team_id,
      })),
    [teams],
  );

  const handlePopupScroll = (e: UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const scrollRatio =
      (target.scrollTop + target.clientHeight) / target.scrollHeight;
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
      notFoundContent={isLoading ? <LoadingOutlined spin /> : "No teams found"}
      options={options}
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
    />
  );
};

export default TeamDropdown;
