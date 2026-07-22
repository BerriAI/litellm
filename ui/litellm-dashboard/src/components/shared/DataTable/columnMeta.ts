import type { RowData } from "@tanstack/react-table";
import type * as React from "react";

import type { ColumnPinnedSide, DataTableSkeletonShape } from "./types";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> {
    numeric?: boolean;
    className?: string;
    headerClassName?: string;
    title?: string;
    pinned?: ColumnPinnedSide;
    skeleton?: DataTableSkeletonShape;
    /** Full control over this column's loading skeleton, for cells the built-in shapes can't mirror. */
    renderSkeleton?: () => React.ReactNode;
  }
}
