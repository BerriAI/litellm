import React from "react";
import { Trash2 } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ACTION_ITEMS, SEVERITY_ITEMS } from "./action_options";

interface ContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface CategoryTableProps {
  categories: ContentCategory[];
  onActionChange?: (id: string, action: "BLOCK" | "MASK") => void;
  onSeverityChange?: (id: string, severity: "high" | "medium" | "low") => void;
  onRemove?: (id: string) => void;
  readOnly?: boolean;
}

const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onActionChange,
  onSeverityChange,
  onRemove,
  readOnly = false,
}) => {
  const columns: ColumnDef<ContentCategory>[] = [
    {
      header: "Category",
      accessorKey: "display_name",
      cell: ({ row }) => {
        const { category, display_name: displayName } = row.original;
        return (
          <div>
            <span className="font-semibold">{displayName}</span>
            {displayName !== category && <div className="text-xs text-muted-foreground">{category}</div>}
          </div>
        );
      },
    },
    {
      header: "Severity Threshold",
      accessorKey: "severity_threshold",
      size: 180,
      cell: ({ row }) => {
        const { id, severity_threshold: severity } = row.original;
        if (readOnly) {
          return <Badge variant={severity === "high" ? "destructive" : "secondary"}>{severity.toUpperCase()}</Badge>;
        }
        return (
          <Select
            items={SEVERITY_ITEMS}
            value={severity}
            onValueChange={(value: string | null) =>
              value && onSeverityChange?.(id, value as "high" | "medium" | "low")
            }
          >
            <SelectTrigger size="sm" className="w-[150px]" aria-label="Severity Threshold">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SEVERITY_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      },
    },
    {
      header: "Action",
      accessorKey: "action",
      size: 150,
      cell: ({ row }) => {
        const { action, id } = row.original;
        if (readOnly) {
          return <Badge variant={action === "BLOCK" ? "destructive" : "secondary"}>{action}</Badge>;
        }
        return (
          <Select
            items={ACTION_ITEMS}
            value={action}
            onValueChange={(value: string | null) => value && onActionChange?.(id, value as "BLOCK" | "MASK")}
          >
            <SelectTrigger size="sm" className="w-[120px]" aria-label="Action">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ACTION_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      },
    },
  ];

  if (!readOnly) {
    columns.push({
      header: "",
      id: "actions",
      size: 100,
      cell: ({ row }) => (
        <Button variant="ghost" size="sm" onClick={() => onRemove?.(row.original.id)}>
          <Trash2 />
          Delete
        </Button>
      ),
    });
  }

  if (categories.length === 0) {
    return <div className="py-10 text-center text-muted-foreground">No categories configured.</div>;
  }

  return <DataTable data={categories} columns={columns} getRowId={(row) => row.id} size="compact" />;
};

export default CategoryTable;
