import { Loader2 } from "lucide-react";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { toast } from "@/lib/toast";
import ExportFormatSelector from "./ExportFormatSelector";
import ExportSummary from "./ExportSummary";
import ExportTypeSelector from "./ExportTypeSelector";
import type { EntityUsageExportModalProps, ExportFormat, ExportType } from "./types";
import { downloadBlob, exportFilename } from "./utils";
import { useLanguage } from "@/contexts/LanguageContext";

const EntityUsageExportModal: React.FC<EntityUsageExportModalProps> = ({
  isOpen,
  onClose,
  entityType,
  onExport,
  dateRange,
  selectedFilters = [],
  customTitle,
}) => {
  const { t } = useLanguage();
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportType, setExportType] = useState<ExportType>("daily");
  const [isExporting, setIsExporting] = useState(false);

  // 实体来自固定枚举；调用方自定义标题不作为翻译键。
  const entityLabel = t(entityType.charAt(0).toUpperCase() + entityType.slice(1));
  const modalTitle = customTitle || t("Export {entity} Usage", { entity: entityLabel });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const blob = await onExport(exportType, exportFormat);
      downloadBlob(blob, exportFilename(entityType, exportType, exportFormat, dateRange));
      toast.success(
        t("{entity} usage data exported successfully as {format}", {
          entity: entityLabel,
          format: exportFormat.toUpperCase(),
        }),
      );
      onClose();
    } catch (error) {
      console.error("Error exporting data:", error);
      toast.fromError(t("Failed to export data"));
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">{modalTitle}</DialogTitle>
        </DialogHeader>
        <div className="space-y-5 py-2">
          <ExportSummary dateRange={dateRange} selectedFilters={selectedFilters} />
          <ExportTypeSelector value={exportType} onChange={setExportType} entityType={entityType} />
          <ExportFormatSelector value={exportFormat} onChange={setExportFormat} />
          <div className="flex items-center justify-end gap-2 pt-4 border-t">
            <Button variant="outline" onClick={onClose} disabled={isExporting}>
              {t("Cancel")}
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting && <Loader2 className="animate-spin" />}
              {isExporting ? t("Exporting...") : t("Export {format}", { format: exportFormat.toUpperCase() })}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EntityUsageExportModal;
