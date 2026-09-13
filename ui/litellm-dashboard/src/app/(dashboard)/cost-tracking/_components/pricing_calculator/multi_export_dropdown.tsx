import React from "react";
import { Download, FileSpreadsheet, FileText } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MultiModelResult } from "./types";
import { exportMultiToPDF, exportMultiToCSV } from "./multi_export_utils";

interface MultiExportDropdownProps {
  multiResult: MultiModelResult;
}

const MultiExportDropdown: React.FC<MultiExportDropdownProps> = ({ multiResult }) => {
  const hasResults = multiResult.entries.some((e) => e.result !== null);

  if (!hasResults) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className={buttonVariants({ variant: "secondary", size: "xs" })}>
        <Download />
        Export
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem onClick={() => exportMultiToPDF(multiResult)}>
          <FileText />
          Export as PDF
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => exportMultiToCSV(multiResult)}>
          <FileSpreadsheet />
          Export as CSV
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default MultiExportDropdown;
