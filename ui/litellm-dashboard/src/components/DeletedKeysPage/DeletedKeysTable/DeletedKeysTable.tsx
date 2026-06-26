"use client";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ChevronDownIcon, ChevronUpIcon, SwitchVerticalIcon } from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  PaginationState,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { Tooltip } from "antd";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { KeyResponse } from "../../key_team_helpers/key_list";

interface DeletedKeysTableProps {
  keys: KeyResponse[];
  totalCount: number;
  isLoading: boolean;
  isFetching: boolean;
  pageIndex: number;
  pageSize: number;
  onPageChange: (pageIndex: number) => void;
}

export function DeletedKeysTable({
  keys,
  totalCount,
  isLoading,
  isFetching,
  pageIndex,
  pageSize,
  onPageChange,
}: DeletedKeysTableProps) {
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>([
    {
      id: "deleted_at",
      desc: true,
    },
  ]);

  const [tablePagination, setTablePagination] = useState<PaginationState>({
    pageIndex,
    pageSize,
  });

  // Sync pagination state when prop changes
  React.useEffect(() => {
    setTablePagination({ pageIndex, pageSize });
  }, [pageIndex, pageSize]);

  const columns: ColumnDef<KeyResponse>[] = [
    {
      id: "token",
      accessorKey: "token",
      header: t("deletedKeys.deletedKeysTable.colKeyId"),
      size: 150,
      maxSize: 250,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value}>
            <span className="font-mono text-blue-500 text-xs truncate block max-w-[250px]">{value || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "key_alias",
      accessorKey: "key_alias",
      header: t("deletedKeys.deletedKeysTable.colKeyAlias"),
      size: 150,
      maxSize: 200,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value}>
            <span className="font-mono text-xs truncate block max-w-[200px]">{value ?? "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "team_alias",
      accessorKey: "team_alias",
      header: t("deletedKeys.deletedKeysTable.colTeamAlias"),
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = info.getValue() as string;
        return <span className="truncate block max-w-[180px]">{value || "-"}</span>;
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: t("deletedKeys.deletedKeysTable.colSpend"),
      size: 100,
      maxSize: 140,
      cell: (info) => (
        <span className="block max-w-[140px]">{formatNumberWithCommas(info.getValue() as number, 4)}</span>
      ),
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: t("deletedKeys.deletedKeysTable.colBudget"),
      size: 110,
      maxSize: 150,
      cell: (info) => {
        const maxBudget = info.getValue() as number | null;
        return (
          <span className="block max-w-[150px]">
            {maxBudget === null ? t("deletedKeys.deletedKeysTable.unlimited") : `$${formatNumberWithCommas(maxBudget)}`}
          </span>
        );
      },
    },
    {
      id: "user_email",
      accessorKey: "user_email",
      header: t("deletedKeys.deletedKeysTable.colUserEmail"),
      size: 160,
      maxSize: 250,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <Tooltip title={value}>
            <span className="font-mono text-xs truncate block max-w-[250px]">{value ?? "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "user_id",
      accessorKey: "user_id",
      header: t("deletedKeys.deletedKeysTable.colUserId"),
      size: 120,
      maxSize: 200,
      cell: (info) => {
        const userId = info.getValue() as string | null;
        return (
          <Tooltip title={userId || undefined}>
            <span className="truncate block max-w-[200px]">{userId || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: t("common.createdAt"),
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = info.getValue();
        return (
          <span className="block max-w-[140px]">{value ? new Date(value as string).toLocaleDateString() : "-"}</span>
        );
      },
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      header: t("deletedKeys.deletedKeysTable.colCreatedBy"),
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = (info.row.original as any).created_by as string | null | undefined;
        return (
          <Tooltip title={value || undefined}>
            <span className="truncate block max-w-[180px]">{value || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "deleted_at",
      accessorKey: "deleted_at",
      header: t("deletedKeys.deletedKeysTable.colDeletedAt"),
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = (info.row.original as any).deleted_at as string | null | undefined;
        return <span className="block max-w-[140px]">{value ? new Date(value).toLocaleDateString() : "-"}</span>;
      },
    },
    {
      id: "deleted_by",
      accessorKey: "deleted_by",
      header: t("deletedKeys.deletedKeysTable.colDeletedBy"),
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = (info.row.original as any).deleted_by as string | null | undefined;
        return (
          <Tooltip title={value || undefined}>
            <span className="truncate block max-w-[180px]">{value || "-"}</span>
          </Tooltip>
        );
      },
    },
  ];

  const table = useReactTable({
    data: keys,
    columns,
    columnResizeMode: "onChange",
    columnResizeDirection: "ltr",
    state: {
      sorting,
      pagination: tablePagination,
    },
    onSortingChange: setSorting,
    onPaginationChange: (updater) => {
      const newPagination = typeof updater === "function" ? updater(tablePagination) : updater;
      setTablePagination(newPagination);
      onPageChange(newPagination.pageIndex);
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    enableSorting: true,
    manualSorting: false,
    manualPagination: true,
    pageCount: Math.ceil(totalCount / pageSize),
  });

  const { pageIndex: currentPageIndex } = table.getState().pagination;
  const start = currentPageIndex * pageSize + 1;
  const end = Math.min((currentPageIndex + 1) * pageSize, totalCount);
  const rangeLabel = `${start} - ${end}`;

  return (
    <div className="w-full h-full overflow-hidden">
      <div className="border-b py-4 flex-1 overflow-hidden">
        <div className="flex items-center justify-between w-full mb-4">
          {isLoading || isFetching ? (
            <span className="inline-flex text-sm text-gray-700">{t("common.loading")}</span>
          ) : (
            <span className="inline-flex text-sm text-gray-700">
              {t("deletedKeys.deletedKeysTable.showingRange", { range: rangeLabel, total: totalCount })}
            </span>
          )}

          <div className="inline-flex items-center gap-2">
            {isLoading || isFetching ? (
              <span className="text-sm text-gray-700">{t("common.loading")}</span>
            ) : (
              <span className="text-sm text-gray-700">
                {t("deletedKeys.deletedKeysTable.pageOf", {
                  current: currentPageIndex + 1,
                  total: table.getPageCount(),
                })}
              </span>
            )}

            <button
              onClick={() => table.previousPage()}
              disabled={isLoading || isFetching || !table.getCanPreviousPage()}
              className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {t("common.previous")}
            </button>

            <button
              onClick={() => table.nextPage()}
              disabled={isLoading || isFetching || !table.getCanNextPage()}
              className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {t("common.next")}
            </button>
          </div>
        </div>
        <div className="h-[75vh] overflow-auto">
          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1" style={{ width: table.getCenterTotalSize() }}>
                <TableHead>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <TableRow key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <TableHeaderCell
                          key={header.id}
                          data-header-id={header.id}
                          className={`py-1 h-8 relative hover:bg-gray-50`}
                          style={{
                            width: header.getSize(),
                            maxWidth: header.column.columnDef.maxSize,
                            position: "relative",
                          }}
                          onMouseEnter={() => {
                            const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                            if (resizer) {
                              (resizer as HTMLElement).style.opacity = "0.5";
                            }
                          }}
                          onMouseLeave={() => {
                            const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                            if (resizer && !header.column.getIsResizing()) {
                              (resizer as HTMLElement).style.opacity = "0";
                            }
                          }}
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center">
                              {header.isPlaceholder
                                ? null
                                : flexRender(header.column.columnDef.header, header.getContext())}
                            </div>
                            <div className="w-4">
                              {header.column.getIsSorted() ? (
                                {
                                  asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                                  desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />,
                                }[header.column.getIsSorted() as string]
                              ) : (
                                <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                              )}
                            </div>
                            <div
                              onDoubleClick={() => header.column.resetSize()}
                              onMouseDown={header.getResizeHandler()}
                              onTouchStart={header.getResizeHandler()}
                              className={`resizer ${table.options.columnResizeDirection} ${header.column.getIsResizing() ? "isResizing" : ""}`}
                              style={{
                                position: "absolute",
                                right: 0,
                                top: 0,
                                height: "100%",
                                width: "5px",
                                background: header.column.getIsResizing() ? "#3b82f6" : "transparent",
                                cursor: "col-resize",
                                userSelect: "none",
                                touchAction: "none",
                                opacity: header.column.getIsResizing() ? 1 : 0,
                              }}
                            />
                          </div>
                        </TableHeaderCell>
                      ))}
                    </TableRow>
                  ))}
                </TableHead>
                <TableBody>
                  {isLoading || isFetching ? (
                    <TableRow>
                      <TableCell colSpan={columns.length} className="h-8 text-center">
                        <div className="text-center text-gray-500">
                          <p>{t("deletedKeys.deletedKeysTable.loadingKeys")}</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : keys.length > 0 ? (
                    table.getRowModel().rows.map((row) => (
                      <TableRow key={row.id} className="h-8">
                        {row.getVisibleCells().map((cell) => (
                          <TableCell
                            key={cell.id}
                            style={{
                              width: cell.column.getSize(),
                              maxWidth: cell.column.columnDef.maxSize,
                              whiteSpace: "pre-wrap",
                              overflow: "hidden",
                            }}
                            className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={columns.length} className="h-8 text-center">
                        <div className="text-center text-gray-500">
                          <p>{t("deletedKeys.deletedKeysTable.noDeletedKeys")}</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
