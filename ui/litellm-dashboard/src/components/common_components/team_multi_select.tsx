import React, { useMemo, useState, type UIEvent } from "react";
import { Select, Typography } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Team } from "../key_team_helpers/key_list";

const { Text } = Typography;

interface TeamMultiSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  disabled?: boolean;
  organizationId?: string | null;
  pageSize?: number;
  placeholder?: string;
}

const SCROLL_THRESHOLD = 0.8;
const DEBOUNCE_MS = 300;

const TeamMultiSelect: React.FC<TeamMultiSelectProps> = ({
  value = [],
  onChange,
  disabled,
  organizationId,
  pageSize = 20,
  placeholder = "Search teams by alias...",
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

  return (
    <Select
      mode="multiple"
      showSearch
      placeholder={placeholder}
      value={value}
      onChange={(val: string[]) => onChange?.(val)}
      disabled={disabled}
      allowClear
      filterOption={false}
      onSearch={handleSearch}
      searchValue={searchInput}
      onPopupScroll={handlePopupScroll}
      loading={isLoading}
      notFoundContent={isLoading ? <LoadingOutlined spin /> : "No teams found"}
      style={{ width: "100%" }}
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
          <span className="font-medium">{team.team_alias}</span>{" "}
          <Text type="secondary">({team.team_id})</Text>
        </Select.Option>
      ))}
    </Select>
  );
};

export default TeamMultiSelect;
