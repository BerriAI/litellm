import TeamMultiSelect from "@/components/common_components/team_multi_select";
import type { EntityType } from "@/components/EntityUsageExport/types";
import { PaginatedMultiSelect } from "@/components/shared/PaginatedMultiSelect";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { Segmented } from "../overview/Primitives";

export type UsageGroupBy = "tag" | "team";

const TAG_FIRST_OPTIONS = [
  { value: "tag", label: "Tag" },
  { value: "team", label: "Team" },
] as const satisfies readonly { value: UsageGroupBy; label: string }[];
const TEAM_FIRST_OPTIONS = [
  { value: "team", label: "Team" },
  { value: "tag", label: "Tag" },
] as const satisfies readonly { value: UsageGroupBy; label: string }[];

interface BreakdownFilterSlotProps {
  entityType: EntityType;
  groupBy: UsageGroupBy;
  onGroupByChange: (groupBy: UsageGroupBy) => void;
  selectedTeamIds: string[];
  onTeamIdsChange: (teamIds: string[]) => void;
  tagOptions: SearchSelectOption[];
  selectedTagFilters: string[];
  onTagFiltersChange: (tags: string[]) => void;
}

export function BreakdownFilterSlot({
  entityType,
  groupBy,
  onGroupByChange,
  selectedTeamIds,
  onTeamIdsChange,
  tagOptions,
  selectedTagFilters,
  onTagFiltersChange,
}: BreakdownFilterSlotProps) {
  const toggle = (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Break down by</span>
      <Segmented
        label="Break down by"
        value={groupBy}
        options={entityType === "team" ? TEAM_FIRST_OPTIONS : TAG_FIRST_OPTIONS}
        onChange={onGroupByChange}
      />
    </div>
  );

  if (entityType === "tag") {
    return (
      <div className="flex items-end gap-3">
        <div>
          <label className="text-sm font-medium text-foreground block mb-2">Filter by team</label>
          <TeamMultiSelect value={selectedTeamIds} onChange={onTeamIdsChange} />
        </div>
        {toggle}
      </div>
    );
  }
  if (entityType === "team") {
    return (
      <div className="flex items-end gap-3">
        <div>
          <label className="text-sm font-medium text-foreground block mb-2">Filter by tag</label>
          <PaginatedMultiSelect
            options={tagOptions}
            value={selectedTagFilters}
            onValueChange={onTagFiltersChange}
            onSearchChange={() => {}}
            onLoadMore={() => {}}
            placeholder="Search tags..."
          />
        </div>
        {toggle}
      </div>
    );
  }
  return null;
}
