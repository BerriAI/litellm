import { FilterInput } from "@/components/common_components/Filters/FilterInput";
import { FiltersButton } from "@/components/common_components/Filters/FiltersButton";
import { ResetFiltersButton } from "@/components/common_components/Filters/ResetFiltersButton";
import { Search, User } from "lucide-react";

interface OrganizationFiltersProps {
  filters: FilterState;
  showFilters: boolean;
  onToggleFilters: (toggle: boolean) => void;
  onChange: <K extends keyof FilterState>(key: K, value: FilterState[K]) => void;
  onReset: () => void;
}

type FilterState = {
  org_id: string;
  org_alias: string;
  sort_by: string;
  sort_order: "asc" | "desc";
};

const OrganizationFilters = ({
  filters,
  showFilters,
  onToggleFilters,
  onChange,
  onReset,
}: OrganizationFiltersProps) => {
  const hasActiveFilters = !!(filters.org_id || filters.org_alias);

  return (
    <div className="flex flex-col space-y-4">
      {/* Search and Filter Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <FilterInput
          placeholder="Search by Organization Name"
          value={filters.org_alias}
          onChange={(value) => onChange("org_alias", value)}
          icon={Search}
          className="w-64"
        />

        <FiltersButton
          onClick={() => onToggleFilters(!showFilters)}
          active={showFilters}
          hasActiveFilters={hasActiveFilters}
        />

        <ResetFiltersButton onClick={onReset} />
      </div>

      {/* Additional Filters */}
      {showFilters && (
        <div className="flex flex-wrap items-center gap-3 mt-3">
          <FilterInput
            placeholder="Search by Organization ID"
            value={filters.org_id}
            onChange={(value) => onChange("org_id", value)}
            icon={User}
            className="w-64"
          />
        </div>
      )}
    </div>
  );
};

export default OrganizationFilters;
export type { FilterState };
