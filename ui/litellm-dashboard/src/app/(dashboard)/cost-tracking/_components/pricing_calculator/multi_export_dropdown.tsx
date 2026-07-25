import React, { useState, useRef, useEffect } from "react";
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

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  if (!hasResults) {
    return null;
  }

  return (
    <div className="relative inline-block" ref={menuRef}>
      <Button size="xs" variant="secondary" onClick={() => setIsOpen(!isOpen)}>
        <Download />
        Export
      </Button>

      {isOpen && (
        <div className="absolute right-0 z-50 mt-1 w-44 rounded-lg border border-border bg-popover py-1 shadow-lg">
          <button
            className="flex w-full items-center px-4 py-2 text-sm text-foreground transition-colors hover:bg-muted"
            onClick={() => {
              exportMultiToPDF(multiResult);
              setIsOpen(false);
            }}
          >
            <FileText className="mr-3 size-4" />
            Export as PDF
          </button>
          <button
            className="flex w-full items-center px-4 py-2 text-sm text-foreground transition-colors hover:bg-muted"
            onClick={() => {
              exportMultiToCSV(multiResult);
              setIsOpen(false);
            }}
          >
            <FileSpreadsheet className="mr-3 size-4" />
            Export as CSV
          </button>
        </div>
      )}
    </div>
  );
};

export default MultiExportDropdown;
