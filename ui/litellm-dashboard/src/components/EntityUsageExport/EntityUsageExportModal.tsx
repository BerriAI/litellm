import React, { useState } from "react";
import { Button } from "@tremor/react";
import { Modal } from "antd";
import Papa from "papaparse";
import NotificationsManager from "../molecules/notifications_manager";
import ExportSummary from "./ExportSummary";
import ExportTypeSelector from "./ExportTypeSelector";
import ExportFormatSelector from "./ExportFormatSelector";
import { generateExportData, generateMetadata } from "./utils";
import type { EntityUsageExportModalProps, ExportFormat, ExportScope } from "./types";

const EntityUsageExportModal: React.FC<EntityUsageExportModalProps> = ({
  isOpen,
  onClose,
  entityType,
  spendData,
  dateRange,
  selectedFilters,
  customTitle,
}) => {
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportScope, setExportScope] = useState<ExportScope>("daily");
  const [isExporting, setIsExporting] = useState(false);

  const entityLabel = entityType === "tag" ? "Tag" : "Team";
  const modalTitle = customTitle || `Export ${entityLabel} Usage`;

  const handleExportCSV = () => {
    const data = generateExportData(spendData, exportScope, entityLabel);
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const fileName = `${entityType}_usage_${exportScope}_${new Date().toISOString().split("T")[0]}.csv`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const data = generateExportData(spendData, exportScope, entityLabel);
    const metadata = generateMetadata(entityType, dateRange, selectedFilters, exportScope, spendData);
    const exportObject = {
      metadata,
      data,
    };
    const jsonString = JSON.stringify(exportObject, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const fileName = `${entityType}_usage_${exportScope}_${new Date().toISOString().split("T")[0]}.json`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const handleExport = async (format?: ExportFormat) => {
    const formatToUse = format || exportFormat;
    setIsExporting(true);
    try {
      if (formatToUse === "csv") {
        handleExportCSV();
        NotificationsManager.success(`${entityLabel} usage data exported successfully as CSV`);
      } else {
        handleExportJSON();
        NotificationsManager.success(`${entityLabel} usage data exported successfully as JSON`);
      }
      onClose();
    } catch (error) {
      console.error("Error exporting data:", error);
      NotificationsManager.fromBackend("Failed to export data");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Modal
      title={<span className="text-base font-semibold">{modalTitle}</span>}
      open={isOpen}
      onCancel={onClose}
      footer={null}
      width={480}
      destroyOnClose
    >
      <div className="space-y-5 py-2">
        <ExportSummary dateRange={dateRange} selectedFilters={selectedFilters} />

        <ExportTypeSelector value={exportScope} onChange={setExportScope} entityType={entityType} />

        <ExportFormatSelector value={exportFormat} onChange={setExportFormat} />

        <div className="flex items-center justify-end gap-2 pt-4 border-t">
          <Button variant="secondary" onClick={onClose} disabled={isExporting} size="sm">
            Cancel
          </Button>
          <Button onClick={() => handleExport()} loading={isExporting} disabled={isExporting} size="sm">
            {isExporting ? "Exporting..." : `Export ${exportFormat.toUpperCase()}`}
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default EntityUsageExportModal;

