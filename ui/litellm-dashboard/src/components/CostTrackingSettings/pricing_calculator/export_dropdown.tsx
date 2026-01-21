import React, { useState, useRef, useEffect } from "react";
import { Button } from "@tremor/react";
import { DownloadOutlined, FilePdfOutlined, FileExcelOutlined } from "@ant-design/icons";
import { CostEstimateResponse } from "../types";
import { exportToPDF, exportToCSV } from "./export_utils";

interface ExportDropdownProps {
  result: CostEstimateResponse;
}

const ExportDropdown: React.FC<ExportDropdownProps> = ({ result }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

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
              exportToPDF(result);
              setIsOpen(false);
            }}
          >
            <FilePdfOutlined className="mr-3 text-red-500" />
            Export as PDF
          </button>
          <button
            className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            onClick={() => {
              exportToCSV(result);
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

export default ExportDropdown;

