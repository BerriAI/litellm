import React from "react";
import TeamDropdown from "./team_dropdown";
import type { FilterOptionCustomComponentProps } from "../molecules/filter";

const FilterTeamDropdown: React.FC<FilterOptionCustomComponentProps> = ({
  value,
  onChange,
}) => <TeamDropdown value={value} onChange={onChange} />;

export default FilterTeamDropdown;
