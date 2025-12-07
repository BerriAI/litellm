import React, { useState } from "react";
import { Button } from "@tremor/react";
import { Modal } from "antd";
import NotificationsManager from "../molecules/notifications_manager";
import ExportSummary from "./ExportSummary";
import ExportTypeSelector from "./ExportTypeSelector";
import ExportFormatSelector from "./ExportFormatSelector";
import { handleExportCSV, handleExportJSON } from "./utils";
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

  const entityLabel = entityType.charAt(0).toUpperCase() + entityType.slice(1);
  const modalTitle = customTitle || `Export ${entityLabel} Usage`;

  const handleExport = async (format?: ExportFormat) => {
    const formatToUse = format || exportFormat;
    setIsExporting(true);
    try {
      if (formatToUse === "csv") {
        handleExportCSV(spendData, exportScope, entityLabel, entityType);
        NotificationsManager.success(`${entityLabel} usage data exported successfully as CSV`);
      } else {
        handleExportJSON(spendData, exportScope, entityLabel, entityType, dateRange, selectedFilters);
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
