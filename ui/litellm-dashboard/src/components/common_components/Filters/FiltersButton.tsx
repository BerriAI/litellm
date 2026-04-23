import { Button } from "@/components/ui/button";
import { Filter } from "lucide-react";
import React from "react";
import { cn } from "@/lib/utils";

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
    <div className="relative inline-flex">
      <Button
        variant="outline"
        onClick={onClick}
        className={cn(active && "bg-muted")}
      >
        <Filter size={16} />
        {label}
      </Button>
      {hasActiveFilters && (
        <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-blue-500" />
      )}
    </div>
  );
};
