"use client";

import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cva.config";

import { SubHead } from "./SectionCard";

export interface LimitTableRow {
  readonly key: string;
  readonly cells: readonly ReactNode[];
}

interface LimitTableProps {
  title: string;
  searchLabel: string;
  columns: readonly string[];
  rows: readonly LimitTableRow[];
  visibleRows?: number;
}

const SEARCH_THRESHOLD = 8;

const GRID_COLUMNS: Readonly<Record<number, string>> = {
  2: "grid-cols-[minmax(0,2fr)_minmax(64px,1fr)] md:grid-cols-[minmax(0,1fr)_140px]",
  3: "grid-cols-[minmax(0,2fr)_minmax(64px,1fr)_minmax(64px,1fr)] md:grid-cols-[minmax(0,1fr)_140px_140px]",
};

export function LimitTable({ title, searchLabel, columns, rows, visibleRows = 5 }: LimitTableProps) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matching = needle ? rows.filter((row) => row.key.toLowerCase().includes(needle)) : rows;
  const shown = expanded ? matching : matching.slice(0, visibleRows);
  const grid = cn("grid items-center gap-2", GRID_COLUMNS[columns.length]);

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <SubHead title={title} count={rows.length} />
      {expanded && rows.length > SEARCH_THRESHOLD ? (
        <Input
          className="h-8 max-w-[280px]"
          placeholder={searchLabel}
          aria-label={searchLabel}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      ) : null}
      <div role="table" aria-label={title} className="flex flex-col">
        <div role="row" className={cn(grid, "min-h-6 text-xs font-medium text-muted-foreground")}>
          {columns.map((column, index) => (
            <span key={column} role="columnheader" className={cn("md:whitespace-nowrap", index > 0 && "text-right")}>
              {column}
            </span>
          ))}
        </div>
        {shown.map((row) => (
          <div key={row.key} role="row" className={cn(grid, "min-h-8 border-t border-border")}>
            {row.cells.map((cell, index) => (
              <span key={index} role="cell" className={cn("min-w-0 truncate tabular-nums", index > 0 && "text-right")}>
                {cell}
              </span>
            ))}
          </div>
        ))}
        {needle && matching.length === 0 ? (
          <div role="row" className="flex min-h-8 items-center border-t border-border text-muted-foreground">
            <span role="cell">{t("keyDetail.overview.limitTable.noMatch", { query })}</span>
          </div>
        ) : null}
      </div>
      {rows.length > visibleRows ? (
        <div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setExpanded(!expanded);
              setQuery("");
            }}
          >
            {expanded
              ? t("keyDetail.overview.limitTable.showFewer")
              : t("keyDetail.overview.limitTable.showAll", { count: rows.length })}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
