"use client";

import {
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { ChevronDownIcon, ChevronUpIcon, SwitchVerticalIcon } from "@heroicons/react/outline";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import React, { useState } from "react";
import { KeyResponse } from "../key_team_helpers/key_list";
import KeyInfoView from "../templates/key_info_view";
import { getVirtualKeysColumns } from "../VirtualKeysPage/virtualKeysColumns";
import { TeamData } from "./TeamInfo";

interface TeamKeysTabProps {
  teamData: TeamData;
  accessToken: string | null;
  onKeyDataUpdate?: () => void;
}

export default function TeamKeysTab({
  teamData,
  accessToken,
  onKeyDataUpdate,
}: TeamKeysTabProps) {
  const [selectedKey, setSelectedKey] = useState<KeyResponse | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  if (!teamData) {
    return null;
  }

  const keys = (teamData.keys ?? []) as KeyResponse[];
  const keysCount = keys.length;

  const teamsForKeyInfo = [
    {
      team_id: teamData.team_id,
      team_alias: teamData.team_info.team_alias,
    },
  ];

  const columns = getVirtualKeysColumns({
    setSelectedKey: (key) => setSelectedKey(key as KeyResponse),
    teams: teamsForKeyInfo,
    expandedAccordions,
    setExpandedAccordions,
  });

  const table = useReactTable({
    data: keys,
    columns,
    columnResizeMode: "onChange",
    columnResizeDirection: "ltr",
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    enableSorting: true,
    initialState: {
      pagination: {
        pageSize: 50,
      },
    },
  });

  if (selectedKey) {
    return (
      <KeyInfoView
        keyId={selectedKey.token ?? selectedKey.token_id ?? ""}
        onClose={() => setSelectedKey(null)}
        keyData={selectedKey}
        teams={teamsForKeyInfo}
        onDelete={onKeyDataUpdate}
        backButtonText="Back to Team Keys"
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-700">
          {keysCount} {keysCount === 1 ? "Key" : "Keys"}
        </span>
      </div>
      <div className="overflow-x-auto">
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
                        className="py-1 h-8 relative hover:bg-gray-50"
                        style={{
                          width: header.getSize(),
                          position: "relative",
                          cursor: header.column.getCanSort() ? "pointer" : "default",
                        }}
                        onClick={header.column.getCanSort() ? header.column.getToggleSortingHandler() : undefined}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center">
                            {header.isPlaceholder
                              ? null
                              : flexRender(header.column.columnDef.header, header.getContext())}
                          </div>
                          {header.column.getCanSort() && (
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
                          )}
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
                {keys.length > 0 ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow key={row.id} className="h-8">
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          style={{
                            width: cell.column.getSize(),
                            maxWidth: "8px",
                            whiteSpace: "pre-wrap",
                            overflow: "hidden",
                          }}
                          className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${cell.column.id === "models" && Array.isArray(cell.getValue()) && (cell.getValue() as string[]).length > 3 ? "px-0" : ""}`}
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
                        <p>No keys in this team</p>
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
  );
}
