import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import { Filter } from "lucide-react";
import React from "react";

interface FiltersButtonProps {
  onClick: () => void;
  active: boolean;
  hasActiveFilters: boolean;
  label?: string;
}

export const FiltersButton: React.FC<FiltersButtonProps> = ({
  onClick,
  active,
  hasActiveFilters,
  label = "Filters",
}) => {
  return (
    <span className="relative inline-flex">
      <Button variant="outline" onClick={onClick} className={cn(active && "bg-muted")}>
        <Filter className="size-4" />
        {label}
      </Button>
      {hasActiveFilters && (
        <sup aria-hidden="true" className="absolute -top-0.5 -right-0.5 size-1.5 rounded-full bg-primary" />
      )}
    </span>
  );
};
