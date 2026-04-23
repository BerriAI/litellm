import { Trash2 } from "lucide-react";
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
import React from "react";

interface Pattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface PatternTableProps {
  patterns: Pattern[];
  onActionChange: (id: string, action: "BLOCK" | "MASK") => void;
  onRemove: (id: string) => void;
}

const PatternTable: React.FC<PatternTableProps> = ({
  patterns,
  onActionChange,
  onRemove,
}) => {
  if (patterns.length === 0) {
    return (
      <div className="text-center py-10 text-muted-foreground">
        No patterns added.
      </div>
    );
  }

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">Type</TableHead>
            <TableHead>Pattern name</TableHead>
            <TableHead>Regex pattern</TableHead>
            <TableHead className="w-[150px]">Action</TableHead>
            <TableHead className="w-[100px]" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {patterns.map((record) => (
            <TableRow key={record.id}>
              <TableCell>
                <Badge
                  className={
                    record.type === "prebuilt"
                      ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                      : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                  }
                >
                  {record.type === "prebuilt" ? "Prebuilt" : "Custom"}
                </Badge>
              </TableCell>
              <TableCell>{record.display_name || record.name}</TableCell>
              <TableCell>
                {record.pattern ? (
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">
                    {record.pattern.substring(0, 40)}...
                  </code>
                ) : (
                  "-"
                )}
              </TableCell>
              <TableCell>
                <Select
                  value={record.action}
                  onValueChange={(v) =>
                    onActionChange(record.id, v as "BLOCK" | "MASK")
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
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  onClick={() => onRemove(record.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default PatternTable;
