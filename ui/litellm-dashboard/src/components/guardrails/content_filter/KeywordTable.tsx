import { Trash2 } from "lucide-react";
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

interface BlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface KeywordTableProps {
  keywords: BlockedWord[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onActionChange: (id: string, field: string, value: any) => void;
  onRemove: (id: string) => void;
}

const KeywordTable: React.FC<KeywordTableProps> = ({
  keywords,
  onActionChange,
  onRemove,
}) => {
  if (keywords.length === 0) {
    return (
      <div className="text-center py-10 text-muted-foreground">
        No keywords added.
      </div>
    );
  }

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Keyword</TableHead>
            <TableHead className="w-[150px]">Action</TableHead>
            <TableHead>Description</TableHead>
            <TableHead className="w-[100px]" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {keywords.map((record) => (
            <TableRow key={record.id}>
              <TableCell>{record.keyword}</TableCell>
              <TableCell>
                <Select
                  value={record.action}
                  onValueChange={(v) => onActionChange(record.id, "action", v)}
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
              <TableCell>{record.description || "-"}</TableCell>
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

export default KeywordTable;
