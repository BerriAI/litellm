import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Trash2 } from "lucide-react";
import React from "react";

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

const severityBadgeClass: Record<string, string> = {
  high: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  medium: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  low: "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
};

const actionBadgeClass: Record<string, string> = {
  BLOCK: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  MASK: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
};

const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onActionChange,
  onSeverityChange,
  onRemove,
  readOnly = false,
}) => {
  if (categories.length === 0) {
    return (
      <div className="text-center py-10 text-muted-foreground">
        No categories configured.
      </div>
    );
  }

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Category</TableHead>
            <TableHead className="w-[180px]">Severity Threshold</TableHead>
            <TableHead className="w-[150px]">Action</TableHead>
            {!readOnly && <TableHead className="w-[100px]" />}
          </TableRow>
        </TableHeader>
        <TableBody>
          {categories.map((record) => (
            <TableRow key={record.id}>
              <TableCell>
                <div>
                  <span className="font-bold">{record.display_name}</span>
                  {record.display_name !== record.category && (
                    <div>
                      <span className="text-xs text-muted-foreground">
                        {record.category}
                      </span>
                    </div>
                  )}
                </div>
              </TableCell>
              <TableCell>
                {readOnly ? (
                  <Badge
                    className={cn(
                      severityBadgeClass[record.severity_threshold],
                    )}
                  >
                    {record.severity_threshold.toUpperCase()}
                  </Badge>
                ) : (
                  <Select
                    value={record.severity_threshold}
                    onValueChange={(v) =>
                      onSeverityChange?.(
                        record.id,
                        v as "high" | "medium" | "low",
                      )
                    }
                  >
                    <SelectTrigger className="w-[150px] h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </TableCell>
              <TableCell>
                {readOnly ? (
                  <Badge className={cn(actionBadgeClass[record.action])}>
                    {record.action}
                  </Badge>
                ) : (
                  <Select
                    value={record.action}
                    onValueChange={(v) =>
                      onActionChange?.(record.id, v as "BLOCK" | "MASK")
                    }
                  >
                    <SelectTrigger className="w-[120px] h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BLOCK">Block</SelectItem>
                      <SelectItem value="MASK">Mask</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </TableCell>
              {!readOnly && (
                <TableCell>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    onClick={() => onRemove?.(record.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete
                  </Button>
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default CategoryTable;
