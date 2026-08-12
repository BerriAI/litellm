import React, { useEffect, useRef, useState } from "react";
import { Download, FileSpreadsheet, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MultiModelResult } from "./types";
import { exportMultiToPDF, exportMultiToCSV } from "./multi_export_utils";

interface MultiExportDropdownProps {
  multiResult: MultiModelResult;
}

const MultiExportDropdown: React.FC<MultiExportDropdownProps> = ({ multiResult }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const hasResults = multiResult.entries.some((e) => e.result !== null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  if (!hasResults) {
    return null;
  }

  return (
    <div ref={menuRef} className="relative inline-block">
      <Button
        size="xs"
        variant="secondary"
        aria-expanded={isOpen}
        aria-haspopup="menu"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Download />
        Export
      </Button>
      {isOpen && (
        <div
          role="menu"
          className="absolute right-0 z-50 mt-1 w-44 rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md"
        >
          <Button
            role="menuitem"
            variant="ghost"
            className="w-full justify-start px-3 font-normal"
            onClick={() => {
              exportMultiToPDF(multiResult);
              setIsOpen(false);
            }}
          >
            <FileText className="text-muted-foreground" />
            Export as PDF
          </Button>
          <Button
            role="menuitem"
            variant="ghost"
            className="w-full justify-start px-3 font-normal"
            onClick={() => {
              exportMultiToCSV(multiResult);
              setIsOpen(false);
            }}
          >
            <FileSpreadsheet className="text-muted-foreground" />
            Export as CSV
          </Button>
        </div>
      )}
    </div>
  );
};

export default MultiExportDropdown;
