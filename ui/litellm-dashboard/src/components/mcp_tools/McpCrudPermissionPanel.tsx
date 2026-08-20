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
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { CrudOp, MCPToolEntry, CRUD_GROUP_META, groupToolsByCrud } from "../../utils/mcpToolCrudClassification";

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
  low: "bg-success/15 text-success",
  medium: "bg-warning/15 text-warning",
  high: "bg-destructive/15 text-destructive font-semibold",
  unknown: "bg-muted text-foreground",
};

const GROUP_BORDER: Record<CrudOp, string> = {
  read: "border-success/20",
  create: "border-info/20",
  update: "border-warning/20",
  delete: "border-destructive/30",
  unknown: "border-border",
};

const GROUP_HEADER_BG: Record<CrudOp, string> = {
  read: "bg-success/10",
  create: "bg-info/10",
  update: "bg-warning/10",
  delete: "bg-destructive/10",
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
            (t) => t.name.toLowerCase().includes(lf) || (t.description ?? "").toLowerCase().includes(lf),
          );
          if (!hasMatch) return null;
        }

        const meta = CRUD_GROUP_META[op];
        const fullyAllowed = isGroupFullyAllowed(op);
        const partial = isGroupPartiallyAllowed(op);
        const isCollapsed = collapsed[op];

        return (
          <div key={op} className={`rounded-lg border ${GROUP_BORDER[op]} overflow-hidden`}>
            {/* Group header */}
            <div className={`flex items-center justify-between px-4 py-3 ${GROUP_HEADER_BG[op]}`}>
              <button
                type="button"
                className="flex items-center gap-2 flex-1 text-left"
                onClick={() => toggleCollapse(op)}
              >
                {isCollapsed ? (
                  <ChevronRightIcon className="w-4 h-4 text-muted-foreground shrink-0" />
                ) : (
                  <ChevronDownIcon className="w-4 h-4 text-muted-foreground shrink-0" />
                )}
                <span className="font-semibold text-foreground text-sm">{meta.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${RISK_BADGE[meta.risk]}`}>
                  {meta.risk === "high"
                    ? "High Risk"
                    : meta.risk === "medium"
                      ? "Medium Risk"
                      : meta.risk === "low"
                        ? "Safe"
                        : "Unclassified"}
                </span>
                <span className="text-xs text-muted-foreground ml-1">
                  {group.filter((t) => effectiveAllowed.has(t.name)).length}/{group.length} allowed
                </span>
              </button>

              {!readOnly && (
                <div className="flex items-center gap-2 ml-4">
                  <p className="text-xs text-muted-foreground">
                    {fullyAllowed ? "All on" : partial ? "Partial" : "All off"}
                  </p>
                  {/* Checkbox supports `indeterminate`; Switch does not. */}
                  <Checkbox
                    aria-label={`Allow all ${meta.label} tools`}
                    checked={fullyAllowed}
                    indeterminate={partial}
                    onCheckedChange={(checked) => toggleGroup(op, checked)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
              )}
            </div>

            {/* Description row */}
            {!isCollapsed && (
              <div className="px-4 pt-2 pb-1 text-xs text-muted-foreground bg-card border-b border-border">
                {meta.description}
              </div>
            )}

            {/* Tool list — searchFilter narrows display only; group toggles still cover all tools */}
            {!isCollapsed && (
              <div className="bg-card divide-y divide-gray-50">
                {group
                  .filter(
                    (t) =>
                      !searchFilter ||
                      t.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
                      (t.description ?? "").toLowerCase().includes(searchFilter.toLowerCase()),
                  )
                  .map((tool) => {
                    const allowed = isToolAllowed(tool.name);
                    return (
                      <div
                        key={tool.name}
                        className={`flex items-start gap-3 px-4 py-2.5 transition-colors hover:bg-accent ${
                          !readOnly ? "cursor-pointer" : ""
                        } ${allowed ? "" : "opacity-60"}`}
                        onClick={() => toggleTool(tool.name)}
                      >
                        <Checkbox
                          aria-label={tool.name}
                          checked={allowed}
                          disabled={readOnly}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-foreground text-sm">{tool.name}</p>
                          {tool.description && (
                            <p className="text-xs text-muted-foreground mt-0.5 leading-snug">{tool.description}</p>
                          )}
                        </div>
                        <span
                          className={`text-xs px-1.5 py-0.5 rounded shrink-0 ${
                            allowed ? "bg-success/15 text-success" : "bg-muted text-muted-foreground"
                          }`}
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
