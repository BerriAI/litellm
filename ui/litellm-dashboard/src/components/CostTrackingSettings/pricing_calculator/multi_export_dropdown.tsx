import React from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Download, FileSpreadsheet, FileText } from "lucide-react";
import { MultiModelResult } from "./types";
import { exportMultiToPDF, exportMultiToCSV } from "./multi_export_utils";

interface MultiExportDropdownProps {
  multiResult: MultiModelResult;
}

const MultiExportDropdown: React.FC<MultiExportDropdownProps> = ({
  multiResult,
}) => {
  const hasResults = multiResult.entries.some((e) => e.result !== null);

  if (!hasResults) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" size="sm">
          <Download className="h-3.5 w-3.5" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem onClick={() => exportMultiToPDF(multiResult)}>
          <FileText className="mr-3 h-4 w-4 text-destructive" />
          Export as PDF
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => exportMultiToCSV(multiResult)}>
          <FileSpreadsheet className="mr-3 h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          Export as CSV
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default MultiExportDropdown;
