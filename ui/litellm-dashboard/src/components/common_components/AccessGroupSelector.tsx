import React from "react";
import { Users } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { useAccessGroups, AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

export interface AccessGroupSelectorProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
}

/**
 * Reusable multi-select selector for access groups.
 *
 * - Displays the **access_group_name** in the dropdown.
 * - Returns an array of **access_group_id** values.
 * - Always multi-select since users can assign multiple access groups.
 */
const AccessGroupSelector: React.FC<AccessGroupSelectorProps> = ({
  value,
  onChange,
  placeholder = "Select access groups",
  disabled = false,
  style,
  className,
  showLabel = false,
  labelText = "Access Group",
}) => {
  const { data: accessGroups, isLoading, isError } = useAccessGroups();

  // ── Loading skeleton ─────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div>
        {showLabel && (
          <p className="mb-2 flex items-center text-sm font-medium text-foreground">
            <Users className="mr-2 size-4" /> {labelText}
          </p>
        )}
        <Skeleton className="h-8 w-full" style={style} />
      </div>
    );
  }

  // ── Build options ────────────────────────────────────────────────────────
  const options: MultiSelectOption[] = (accessGroups ?? []).map((group: AccessGroupResponse) => ({
    label: group.access_group_name,
    value: group.access_group_id,
    description: group.access_group_id,
  }));

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div>
      {showLabel && (
        <p className="mb-2 flex items-center text-sm font-medium text-foreground">
          <Users className="mr-2 size-4" /> {labelText}
        </p>
      )}
      <div style={style}>
        <MultiSelect
          options={options}
          value={value}
          onValueChange={onChange ?? (() => {})}
          placeholder={placeholder}
          emptyText={isError ? "Failed to load access groups" : "No access groups found"}
          disabled={disabled}
          className={`w-full rounded-md ${className ?? ""}`}
        />
      </div>
    </div>
  );
};

export default AccessGroupSelector;
