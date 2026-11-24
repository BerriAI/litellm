import type { ComponentProps, ReactNode } from "react";
import type { SortingState } from "@tanstack/react-table";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { SortableTableHead } from "@/components/shared/DataTable/SortableTableHead";
import { cn } from "@/lib/cva.config";

export function CustomerUsageTable({
  label,
  headers,
  columnCount,
  message,
  children,
}: {
  label: string;
  headers: ReactNode;
  columnCount: number;
  message?: string;
  children: ReactNode;
}) {
  return (
    <Table aria-label={label}>
      <TableHeader>
        <TableRow>{headers}</TableRow>
      </TableHeader>
      <TableBody>
        {message !== undefined ? (
          <TableRow>
            <TableCell colSpan={columnCount} className="py-8 text-center text-muted-foreground">
              {message}
            </TableCell>
          </TableRow>
        ) : (
          children
        )}
      </TableBody>
    </Table>
  );
}

export function CustomerUsageTableHeaders({
  columns,
  sorting,
  onSort,
}: {
  columns: readonly { id: string; label: ParseKeys; defaultDescending?: boolean }[];
  sorting: SortingState;
  onSort: (sorting: SortingState) => void;
}) {
  const { t } = useTranslation();
  return columns.map(({ id, label, defaultDescending }) => {
    const rightAligned = id === "status" || id === "tokens" || id === "actions";
    const className = cn(
      "px-4",
      id === "spend" && "w-[30%] min-w-80",
      rightAligned && "text-right",
      id === "tokens" && defaultDescending === undefined && "whitespace-normal",
    );
    return defaultDescending === undefined ? (
      <TableHead key={id} className={className}>
        {t(label)}
      </TableHead>
    ) : (
      <SortableTableHead
        key={id}
        id={id}
        sorting={sorting}
        onSort={onSort}
        defaultDescending={defaultDescending}
        className={className}
        buttonClassName={id === "share" ? "h-auto min-h-8 py-1 text-left whitespace-normal" : undefined}
      >
        {t(label)}
      </SortableTableHead>
    );
  });
}

export function CustomerUsageEditButton({
  children,
  ...props
}: Pick<ComponentProps<typeof Button>, "children" | "onClick" | "disabled" | "aria-label" | "type">) {
  return (
    <Button variant="outline" size="sm" {...props}>
      <Pencil aria-hidden="true" className="size-3.5" />
      {children}
    </Button>
  );
}
