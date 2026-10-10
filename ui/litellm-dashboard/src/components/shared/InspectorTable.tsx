"use client";

import "@/components/shared/DataTable/columnMeta";

import { flexRender, type Row as TanStackRow, type Table as TanStackTable } from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronRight } from "lucide-react";
import { createContext, Fragment, useContext, useState, type ComponentProps, type ReactNode } from "react";

import { Inspector } from "@/components/shared/Inspector";
import { Skeleton } from "@/components/ui/skeleton";
import { TableBody, TableCell, TableHead, TableHeader } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";

const OVERSCAN_ROWS = 12;
const NUMERIC = "text-right tabular-nums";
const ROW =
  "cursor-pointer border-b border-border/60 transition-colors duration-150 hover:bg-trace-row-hover focus-visible:outline-2 focus-visible:outline-ring data-[state=selected]:bg-trace-row-selected data-[state=selected]:shadow-[inset_2px_0_0_var(--trace-brand)] data-[state=selected]:hover:bg-trace-row-selected motion-reduce:transition-none";

interface InspectorTableState<T> {
  readonly table: TanStackTable<T>;
  readonly scroller: HTMLDivElement | null;
}

const InspectorTableContext = createContext<unknown>(null);

export function useInspectorTable<T>(): InspectorTableState<T> {
  const state = useContext(InspectorTableContext);
  if (state === null) throw new Error("InspectorTable parts must be rendered inside InspectorTable.Root");
  return state as InspectorTableState<T>;
}

type RootProps<T> = ComponentProps<"div"> & { readonly table: TanStackTable<T> };

function Root<T>({ table, className, children, ...props }: RootProps<T>) {
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  return (
    <InspectorTableContext.Provider value={{ table, scroller }}>
      <div
        ref={setScroller}
        data-slot="inspector-table"
        className={cn("min-h-0 flex-1 overflow-auto", className)}
        {...props}
      >
        {children}
      </div>
    </InspectorTableContext.Provider>
  );
}

/** A column without `size` takes the remaining width. */
function Grid({ className, children, ...props }: ComponentProps<"table">) {
  const { table } = useInspectorTable();
  return (
    <table data-slot="table" className={cn("w-full table-fixed border-collapse text-left", className)} {...props}>
      <colgroup>
        {table.getVisibleLeafColumns().map((column) => (
          <col key={column.id} style={{ width: column.columnDef.size }} />
        ))}
      </colgroup>
      {children}
    </table>
  );
}

function Header({ hidden = false }: { readonly hidden?: boolean }) {
  const { table } = useInspectorTable();
  return (
    <TableHeader
      className={hidden ? "sr-only" : "sticky top-0 z-sticky bg-[color-mix(in_oklab,var(--muted)_40%,var(--card))]"}
    >
      {table.getHeaderGroups().map((group) => (
        <tr key={group.id} className="h-8 border-border text-xs tracking-wider text-muted-foreground uppercase">
          {group.headers.map(({ id, column, isPlaceholder, getContext }) => (
            <TableHead
              key={id}
              className={cn(
                "h-auto px-3 text-muted-foreground",
                column.columnDef.meta?.numeric && NUMERIC,
                column.columnDef.meta?.headerClassName,
              )}
            >
              {isPlaceholder ? null : flexRender(column.columnDef.header, getContext())}
            </TableHead>
          ))}
        </tr>
      ))}
    </TableHeader>
  );
}

interface BodyProps<T> {
  readonly rowHeight: (row: TanStackRow<T>) => number;
  readonly children: (row: TanStackRow<T>) => ReactNode;
  readonly after?: ReactNode;
  readonly className?: string;
}

function Body<T>({ rowHeight, children, after, className }: BodyProps<T>) {
  const { table, scroller } = useInspectorTable<T>();
  const rows = table.getRowModel().rows;
  const virtualizerOptions = {
    count: rows.length,
    getScrollElement: () => scroller,
    estimateSize: (index: number) => rowHeight(rows[index]),
    overscan: OVERSCAN_ROWS,
    getItemKey: (index: number) => rows[index].id,
  };
  const virtualizer = useVirtualizer(virtualizerOptions);
  const items = virtualizer.getVirtualItems();
  const padTop = items[0]?.start ?? 0;
  const padBottom = virtualizer.getTotalSize() - (items.at(-1)?.end ?? 0);
  return (
    <TableBody className={className}>
      {padTop > 0 && <tr aria-hidden style={{ height: padTop }} />}
      {items.map(({ key, index }) => (
        <Fragment key={key}>{children(rows[index])}</Fragment>
      ))}
      {padBottom > 0 && <tr aria-hidden style={{ height: padBottom }} />}
      {after}
    </TableBody>
  );
}

type RowProps<T> = ComponentProps<"tr"> & { readonly row: TanStackRow<T>; readonly item: unknown };

function Row<T>({ row, item, className, ...props }: RowProps<T>) {
  return (
    <Inspector.Row item={item} render={<tr data-slot="table-row" className={cn(ROW, className)} {...props} />}>
      {row.getVisibleCells().map(({ id, column, getContext }) => (
        <TableCell
          key={id}
          className={cn("px-3 py-0", column.columnDef.meta?.numeric && NUMERIC, column.columnDef.meta?.className)}
        >
          {flexRender(column.columnDef.cell, getContext())}
        </TableCell>
      ))}
    </Inspector.Row>
  );
}

const SKELETON_WIDTHS = ["w-[58%]", "w-[44%]", "w-[70%]", "w-[50%]", "w-[64%]", "w-[48%]"] as const;

type SkeletonRowProps = ComponentProps<"tr"> & { readonly index: number };

/** One placeholder row shaped by the visible columns: `meta.renderSkeleton` wins, numeric cells right-align. */
function SkeletonRow({ index, className, ...props }: SkeletonRowProps) {
  const { table } = useInspectorTable();
  return (
    <tr aria-hidden data-slot="table-skeleton-row" className={cn("border-b border-border/60", className)} {...props}>
      {table.getVisibleLeafColumns().map((column, position) => {
        const meta = column.columnDef.meta;
        return (
          <TableCell key={column.id} className={cn("px-3 py-0", meta?.numeric && NUMERIC, meta?.className)}>
            {meta?.renderSkeleton ? (
              meta.renderSkeleton()
            ) : (
              <Skeleton
                className={cn(
                  "h-3",
                  SKELETON_WIDTHS[(index + position) % SKELETON_WIDTHS.length],
                  meta?.numeric && "ml-auto",
                )}
              />
            )}
          </TableCell>
        );
      })}
    </tr>
  );
}

interface IndentProps<T> {
  readonly row: TanStackRow<T>;
  readonly toggleLabel?: (expanded: boolean) => string;
  readonly className?: string;
}

function Indent<T>({ row, toggleLabel = (expanded) => (expanded ? "Collapse" : "Expand"), className }: IndentProps<T>) {
  if (row.depth > 0) {
    const last = row.getParentRow()?.subRows.at(-1)?.id === row.id;
    return (
      <span aria-hidden="true" className={cn("relative w-6 shrink-0", className)}>
        <span className={cn("absolute top-0 left-3 w-px bg-border", last ? "h-1/2" : "h-full")} />
        <span className="absolute top-1/2 left-3 h-px w-3 bg-border" />
      </span>
    );
  }
  if (!row.getCanExpand()) return <span aria-hidden="true" className="size-6 shrink-0" />;
  const expanded = row.getIsExpanded();
  return (
    <button
      type="button"
      aria-expanded={expanded}
      aria-label={toggleLabel(expanded)}
      onClick={(event) => {
        event.stopPropagation();
        row.toggleExpanded();
      }}
      className="inline-flex size-6 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
    >
      <ChevronRight
        className={cn(
          "size-3.5 transition-transform duration-150 motion-reduce:transition-none",
          expanded && "rotate-90",
        )}
      />
    </button>
  );
}

export const InspectorTable = { Root, Grid, Header, Body, Row, SkeletonRow, Indent } as const;
