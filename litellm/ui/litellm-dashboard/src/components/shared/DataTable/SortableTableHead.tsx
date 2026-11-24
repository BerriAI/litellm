import type { SortingState } from "@tanstack/react-table";
import type { ReactNode } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TableHead } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";

export function SortableTableHead({
  id,
  sorting,
  onSort,
  defaultDescending = false,
  children,
  className,
  buttonClassName,
}: {
  id: string;
  sorting: SortingState;
  onSort: (sorting: SortingState) => void;
  defaultDescending?: boolean;
  children: ReactNode;
  className?: string;
  buttonClassName?: string;
}) {
  const active = sorting[0]?.id === id;
  const descending = sorting[0]?.desc ?? false;
  const SortIcon = descending ? ArrowDown : ArrowUp;
  const Icon = active ? SortIcon : ArrowUpDown;
  const direction = descending ? "descending" : "ascending";
  return (
    <TableHead aria-sort={active ? direction : "none"} className={className}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className={cn("-mx-2 gap-2", buttonClassName)}
        onClick={() => onSort([{ id, desc: active ? !descending : defaultDescending }])}
      >
        {children}
        <Icon aria-hidden="true" className="size-3.5" />
      </Button>
    </TableHead>
  );
}
