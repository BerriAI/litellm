import type { RowData } from "@tanstack/react-table";
import type * as React from "react";

import type { ColumnPinnedSide, DataTableSkeletonShape } from "./types";

declare module "@tanstack/react-table" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- declaration merging requires the type parameters to match the upstream ColumnMeta signature exactly (TS2428)
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
