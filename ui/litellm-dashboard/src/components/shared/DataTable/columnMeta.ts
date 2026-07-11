import type { RowData } from "@tanstack/react-table";

import type { ColumnPinnedSide, DataTableSkeletonShape } from "./types";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> {
    numeric?: boolean;
    className?: string;
    headerClassName?: string;
    title?: string;
    pinned?: ColumnPinnedSide;
    skeleton?: DataTableSkeletonShape;
  }
}
