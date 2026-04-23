import React from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ArrowUpDown, ChevronDown, ChevronUp, X } from "lucide-react";
import { cn } from "@/lib/utils";

export type SortState = "asc" | "desc" | false;

interface TableHeaderSortDropdownProps {
  sortState: SortState;
  onSortChange: (newState: SortState) => void;
  columnId?: string;
}

export const TableHeaderSortDropdown: React.FC<
  TableHeaderSortDropdownProps
> = ({ sortState, onSortChange }) => {
  const renderIcon = () => {
    if (sortState === "asc") return <ChevronUp className="h-4 w-4" />;
    if (sortState === "desc") return <ChevronDown className="h-4 w-4" />;
    return <ArrowUpDown className="h-4 w-4" />;
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={(e) => e.stopPropagation()}
          className={cn(
            "h-6 w-6",
            sortState
              ? "text-primary hover:text-primary"
              : "text-muted-foreground hover:text-primary",
          )}
        >
          {renderIcon()}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => onSortChange("asc")}>
          <ChevronUp className="mr-2 h-4 w-4" />
          Ascending
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSortChange("desc")}>
          <ChevronDown className="mr-2 h-4 w-4" />
          Descending
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onSortChange(false)}>
          <X className="mr-2 h-4 w-4" />
          Reset
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
