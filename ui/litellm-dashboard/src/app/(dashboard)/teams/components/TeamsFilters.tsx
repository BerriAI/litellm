import { Select, SelectItem } from "@tremor/react";
import React from "react";
import { Organization } from "@/components/networking";

interface TeamsFiltersProps {
  filters: FilterState;
  organizations: Organization[] | null;
  showFilters: boolean;
  onToggleFilters: (toggle: boolean) => void;
  onChange: <K extends keyof FilterState>(key: K, value: FilterState[K]) => void;
  onReset: () => void;
}

type FilterState = {
  team_id: string;
  team_alias: string;
  organization_id: string;
  sort_by: string;
  sort_order: "asc" | "desc";
};

const TeamsFilters = ({
  filters,
  organizations,
  showFilters,
  onToggleFilters,
  onChange,
  onReset,
}: TeamsFiltersProps) => {
  return (
    <div className="flex flex-col space-y-4">
      {/* Search and Filter Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Team Alias Search */}
        <div className="relative w-64">
          <input
            type="text"
            placeholder="Search by Team Name..."
            className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={filters.team_alias}
            onChange={(e) => onChange("team_alias", e.target.value)}
          />
          <svg
            className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        {/* Filter Button */}
        <button
          className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? "bg-gray-100" : ""}`}
          onClick={() => onToggleFilters(!showFilters)}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
            />
          </svg>
          Filters
          {(filters.team_id || filters.team_alias || filters.organization_id) && (
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
          )}
        </button>

        {/* Reset Filters Button */}
        <button
          className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
          onClick={onReset}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          Reset Filters
        </button>
      </div>

      {/* Additional Filters */}
      {showFilters && (
        <div className="flex flex-wrap items-center gap-3 mt-3">
          {/* Team ID Search */}
          <div className="relative w-64">
            <input
              type="text"
              placeholder="Enter Team ID"
              className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={filters.team_id}
              onChange={(e) => onChange("team_id", e.target.value)}
            />
            <svg
              className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>

          {/* Organization Dropdown */}
          <div className="w-64">
            <Select
              value={filters.organization_id || ""}
              onValueChange={(value) => onChange("organization_id", value)}
              placeholder="Select Organization"
            >
              {organizations?.map((org) => (
                <SelectItem key={org.organization_id} value={org.organization_id || ""}>
                  {org.organization_alias || org.organization_id}
                </SelectItem>
              ))}
            </Select>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamsFilters;
