import React from "react";
import { Select } from "antd";
import { Team } from "../key_team_helpers/key_list";

interface TeamDropdownProps {
  teams?: Team[] | null;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
}

const TeamDropdown: React.FC<TeamDropdownProps> = ({ teams, value, onChange, disabled }) => {
  console.log("disabled", disabled);
  return (
    <Select
      showSearch
      placeholder="Search or select a team"
      value={value}
      onChange={onChange}
      disabled={disabled}
      allowClear
      filterOption={(input, option) => {
        if (!option) return false;
        // Get team data from the option key
        const team = teams?.find((t) => t.team_id === option.key);
        if (!team) return false;

        const searchTerm = input.toLowerCase().trim();
        const teamAlias = (team.team_alias || "").toLowerCase();
        const teamId = (team.team_id || "").toLowerCase();

        // Search in both team alias and team ID
        return teamAlias.includes(searchTerm) || teamId.includes(searchTerm);
      }}
      optionFilterProp="children"
    >
      {teams?.map((team) => (
        <Select.Option key={team.team_id} value={team.team_id}>
          <span className="font-medium">{team.team_alias}</span> <span className="text-gray-500">({team.team_id})</span>
        </Select.Option>
      ))}
    </Select>
  );
};

export default TeamDropdown;
