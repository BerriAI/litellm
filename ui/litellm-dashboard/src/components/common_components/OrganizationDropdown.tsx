import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Organization } from "../networking";
import { cn } from "@/lib/utils";

interface OrganizationDropdownProps {
  organizations?: Organization[] | null;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  loading?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

const ALL = "__all__";

const OrganizationDropdown: React.FC<OrganizationDropdownProps> = ({
  organizations,
  value,
  onChange,
  disabled,
  loading,
  style,
  className,
}) => {
  return (
    <Select
      value={value ?? ALL}
      onValueChange={(v) => onChange?.(v === ALL ? "" : v)}
      disabled={disabled || loading}
    >
      <SelectTrigger
        style={{ minWidth: 280, ...style }}
        className={cn(className)}
      >
        <SelectValue placeholder="All Organizations" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>All Organizations</SelectItem>
        {organizations?.map((org) => (
          <SelectItem
            key={org.organization_id}
            value={org.organization_id ?? ""}
          >
            <span className="font-medium">{org.organization_alias}</span>{" "}
            <span className="text-muted-foreground">
              ({org.organization_id})
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default OrganizationDropdown;
