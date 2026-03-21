import React, { useState, useRef, useEffect } from "react";
import { Button } from "@tremor/react";
import { DownloadOutlined, FilePdfOutlined, FileExcelOutlined } from "@ant-design/icons";
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
      <Button
        size="xs"
        variant="secondary"
        icon={DownloadOutlined}
        onClick={() => setIsOpen(!isOpen)}
      >
        Export
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-44 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
          <button
            className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            onClick={() => {
              exportMultiToPDF(multiResult);
              setIsOpen(false);
            }}
          >
            <FilePdfOutlined className="mr-3 text-red-500" />
            Export as PDF
          </button>
          <button
            className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            onClick={() => {
              exportMultiToCSV(multiResult);
              setIsOpen(false);
            }}
          >
            <FileExcelOutlined className="mr-3 text-green-600" />
            Export as CSV
          </button>
        </div>
      )}
    </div>
  );
};

export default MultiExportDropdown;

