/**
 * McpCrudPermissionPanel
 *
 * Displays MCP tools grouped by CRUD operation risk category.
 * Lets admins toggle an entire category (Read / Create / Update / Delete)
 * or individual tools within a category.
 *
 * The component is a drop-in replacement for a flat tool checkbox list.
 * Output is the same `string[]` of allowed tool names that the backend accepts.
 */

import React, { useMemo, useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  CrudOp,
  MCPToolEntry,
  CRUD_GROUP_META,
  groupToolsByCrud,
} from "../../utils/mcpToolCrudClassification";

interface McpCrudPermissionPanelProps {
  /** List of tools available on this MCP server. */
  tools: MCPToolEntry[];
  /**
   * Currently allowed tool names.
   * `undefined` means "allow all" (no restriction stored yet).
   * An empty array means "allow none".
   */
  value: string[] | undefined;
  /** Called whenever the allowed set changes. Always emits a concrete string[]. */
  onChange: (allowed: string[]) => void;
  readOnly?: boolean;
  /**
   * Optional search filter string. When set, only tools whose name or description
   * contain this string (case-insensitive) are shown. Group-level toggles still
   * operate on the complete group — not just the visible (filtered) subset.
   */
  searchFilter?: string;
}

const CRUD_ORDER: CrudOp[] = ["read", "create", "update", "delete", "unknown"];

const RISK_BADGE: Record<string, string> = {
  low: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  medium:
    "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  high: "bg-red-100 text-red-800 font-semibold dark:bg-red-950 dark:text-red-300",
  unknown: "bg-muted text-muted-foreground",
};

const GROUP_BORDER: Record<CrudOp, string> = {
  read: "border-emerald-200 dark:border-emerald-900",
  create: "border-blue-200 dark:border-blue-900",
  update: "border-amber-200 dark:border-amber-900",
  delete: "border-red-300 dark:border-red-900",
  unknown: "border-border",
};

const GROUP_HEADER_BG: Record<CrudOp, string> = {
  read: "bg-emerald-50 dark:bg-emerald-950/30",
  create: "bg-blue-50 dark:bg-blue-950/30",
  update: "bg-amber-50 dark:bg-amber-950/30",
  delete: "bg-red-50 dark:bg-red-950/30",
  unknown: "bg-muted",
};

// ---------------------------------------------------------------------------

const McpCrudPermissionPanel: React.FC<McpCrudPermissionPanelProps> = ({
  tools,
  value,
  onChange,
  readOnly = false,
  searchFilter = "",
}) => {
  const [collapsed, setCollapsed] = useState<Record<CrudOp, boolean>>({
    read: false,
    create: false,
    update: false,
    delete: false,
    unknown: true,
  });

  const grouped = useMemo(() => groupToolsByCrud(tools), [tools]);

  /**
   * Derive the effective allowed set:
   * - `undefined` → all tools allowed
   * - We materialise it to a Set<string> for fast lookups.
   */
  const effectiveAllowed: Set<string> = useMemo(() => {
    if (value === undefined) {
      return new Set(tools.map((t) => t.name));
    }
    return new Set(value);
  }, [value, tools]);

  const isToolAllowed = (name: string) => effectiveAllowed.has(name);

  const isGroupFullyAllowed = (op: CrudOp) => {
    const group = grouped[op];
    return group.length > 0 && group.every((t) => effectiveAllowed.has(t.name));
  };

  const isGroupPartiallyAllowed = (op: CrudOp) => {
    const group = grouped[op];
    if (group.length === 0) return false;
    const allowedCount = group.filter((t) => effectiveAllowed.has(t.name)).length;
    return allowedCount > 0 && allowedCount < group.length;
  };

  const toggleTool = (toolName: string) => {
    if (readOnly) return;
    const next = new Set(effectiveAllowed);
    if (next.has(toolName)) {
      next.delete(toolName);
    } else {
      next.add(toolName);
    }
    onChange(Array.from(next));
  };

  const toggleGroup = (op: CrudOp, enable: boolean) => {
    if (readOnly) return;
    const next = new Set(effectiveAllowed);
    for (const tool of grouped[op]) {
      if (enable) {
        next.add(tool.name);
      } else {
        next.delete(tool.name);
      }
    }
    onChange(Array.from(next));
  };

  const toggleCollapse = (op: CrudOp) => {
    setCollapsed((prev) => ({ ...prev, [op]: !prev[op] }));
  };

  if (tools.length === 0) return null;

  return (
    <div className="space-y-3">
      {CRUD_ORDER.map((op) => {
        const group = grouped[op];
        if (group.length === 0) return null;

        // If a search filter is active and no tools in this group match, hide the
        // entire group — including its header — to avoid empty visual blocks.
        if (searchFilter) {
          const lf = searchFilter.toLowerCase();
          const hasMatch = group.some(
            (t) =>
              t.name.toLowerCase().includes(lf) ||
              (t.description ?? "").toLowerCase().includes(lf)
          );
          if (!hasMatch) return null;
        }

        const meta = CRUD_GROUP_META[op];
        const fullyAllowed = isGroupFullyAllowed(op);
        const partial = isGroupPartiallyAllowed(op);
        const isCollapsed = collapsed[op];

        return (
          <div
            key={op}
            className={cn(
              "rounded-lg border overflow-hidden",
              GROUP_BORDER[op],
            )}
          >
            {/* Group header */}
            <div
              className={cn(
                "flex items-center justify-between px-4 py-3",
                GROUP_HEADER_BG[op],
              )}
            >
              <button
                type="button"
                className="flex items-center gap-2 flex-1 text-left"
                onClick={() => toggleCollapse(op)}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                )}
                <span className="font-semibold text-foreground text-sm">
                  {meta.label}
                </span>
                <span
                  className={cn(
                    "text-xs px-2 py-0.5 rounded-full",
                    RISK_BADGE[meta.risk],
                  )}
                >
                  {meta.risk === "high"
                    ? "High Risk"
                    : meta.risk === "medium"
                      ? "Medium Risk"
                      : meta.risk === "low"
                        ? "Safe"
                        : "Unclassified"}
                </span>
                <span className="text-xs text-muted-foreground ml-1">
                  {group.filter((t) => effectiveAllowed.has(t.name)).length}/
                  {group.length} allowed
                </span>
              </button>

              {!readOnly && (
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-xs text-muted-foreground">
                    {fullyAllowed ? "All on" : partial ? "Partial" : "All off"}
                  </span>
                  <Checkbox
                    checked={
                      partial ? "indeterminate" : fullyAllowed ? true : false
                    }
                    onCheckedChange={(c) => toggleGroup(op, c === true)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
              )}
            </div>

            {/* Description row */}
            {!isCollapsed && (
              <div className="px-4 pt-2 pb-1 text-xs text-muted-foreground bg-background border-b border-border">
                {meta.description}
              </div>
            )}

            {/* Tool list */}
            {!isCollapsed && (
              <div className="bg-background divide-y divide-border">
                {group
                  .filter(
                    (t) =>
                      !searchFilter ||
                      t.name
                        .toLowerCase()
                        .includes(searchFilter.toLowerCase()) ||
                      (t.description ?? "")
                        .toLowerCase()
                        .includes(searchFilter.toLowerCase()),
                  )
                  .map((tool) => {
                    const allowed = isToolAllowed(tool.name);
                    return (
                      <div
                        key={tool.name}
                        className={cn(
                          "flex items-start gap-3 px-4 py-2.5 transition-colors hover:bg-muted",
                          !readOnly && "cursor-pointer",
                          !allowed && "opacity-60",
                        )}
                        onClick={() => toggleTool(tool.name)}
                      >
                        <Checkbox
                          checked={allowed}
                          onCheckedChange={() => toggleTool(tool.name)}
                          disabled={readOnly}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-foreground text-sm">
                            {tool.name}
                          </p>
                          {tool.description && (
                            <p className="text-xs text-muted-foreground mt-0.5 leading-snug">
                              {tool.description}
                            </p>
                          )}
                        </div>
                        <span
                          className={cn(
                            "text-xs px-1.5 py-0.5 rounded flex-shrink-0",
                            allowed
                              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                              : "bg-muted text-muted-foreground",
                          )}
                        >
                          {allowed ? "on" : "off"}
                        </span>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default McpCrudPermissionPanel;
