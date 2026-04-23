import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Filter, RotateCcw, Search, User } from "lucide-react";
import { cn } from "@/lib/utils";
import React from "react";
import { Organization } from "@/components/networking";

interface TeamsFiltersProps {
  filters: FilterState;
  organizations: Organization[] | null;
  showFilters: boolean;
  onToggleFilters: (toggle: boolean) => void;
  onChange: <K extends keyof FilterState>(
    key: K,
    value: FilterState[K],
  ) => void;
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
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative w-64">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder="Search by Team Name..."
            className="pl-8"
            value={filters.team_alias}
            onChange={(e) => onChange("team_alias", e.target.value)}
          />
        </div>

        <Button
          variant="outline"
          className={cn(showFilters && "bg-muted")}
          onClick={() => onToggleFilters(!showFilters)}
        >
          <Filter className="h-4 w-4" />
          Filters
          {(filters.team_id || filters.team_alias || filters.organization_id) && (
            <span
              data-testid="active-filter-indicator"
              className="w-2 h-2 rounded-full bg-blue-500"
            />
          )}
        </Button>

        <Button variant="outline" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          Reset Filters
        </Button>
      </div>

      {showFilters && (
        <div className="flex flex-wrap items-center gap-3 mt-3">
          <div className="relative w-64">
            <User className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Enter Team ID"
              className="pl-8"
              value={filters.team_id}
              onChange={(e) => onChange("team_id", e.target.value)}
            />
          </div>

          <div className="w-64">
            <Select
              value={filters.organization_id || undefined}
              onValueChange={(value) => onChange("organization_id", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select Organization" />
              </SelectTrigger>
              <SelectContent>
                {organizations?.map((org) => (
                  <SelectItem
                    key={org.organization_id}
                    value={org.organization_id || ""}
                  >
                    {org.organization_alias || org.organization_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamsFilters;
