import type { ParseKeys } from "i18next";
import { Loader2 } from "lucide-react";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import ExportFormatSelector from "./ExportFormatSelector";
import ExportSummary from "./ExportSummary";
import ExportTypeSelector from "./ExportTypeSelector";
import type { EntityType, EntityUsageExportModalProps, ExportFormat, ExportType } from "./types";
import { downloadBlob, exportFilename } from "./utils";

const ENTITY_LABEL_KEYS: Record<EntityType, ParseKeys> = {
  user: "usage.export.modal.entities.user",
  team: "usage.export.modal.entities.team",
  tag: "usage.export.modal.entities.tag",
  organization: "usage.export.modal.entities.organization",
  customer: "usage.export.modal.entities.customer",
  agent: "usage.export.modal.entities.agent",
};

const EntityUsageExportModal: React.FC<EntityUsageExportModalProps> = ({
  isOpen,
  onClose,
  entityType,
  onExport,
  dateRange,
  selectedFilters = [],
  customTitle,
  canIncludeMetadata,
}) => {
  const { t } = useTranslation();
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportType, setExportType] = useState<ExportType>("daily");
  const [includeMetadata, setIncludeMetadata] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const entityLabel = t(ENTITY_LABEL_KEYS[entityType]);
  const modalTitle = customTitle || t("usage.export.modal.title", { entity: entityLabel });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const blob = await onExport(exportType, exportFormat, includeMetadata);
      downloadBlob(blob, exportFilename(entityType, exportType, exportFormat, dateRange));
      toast.success(t("usage.export.modal.success", { entity: entityLabel, format: exportFormat.toUpperCase() }));
      onClose();
    } catch (error) {
      console.error("Error exporting data:", error);
      toast.fromError(t("usage.export.modal.failed"));
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
          {canIncludeMetadata && (
            <div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="usage-export-include-metadata"
                  checked={includeMetadata}
                  onCheckedChange={(checked) => setIncludeMetadata(checked === true)}
                  aria-describedby="usage-export-include-metadata-note"
                />
                <Label htmlFor="usage-export-include-metadata">{t("usage.export.modal.includeMetadata")}</Label>
              </div>
              <p id="usage-export-include-metadata-note" className="text-xs text-muted-foreground mt-1 ml-6">
                {t("usage.export.modal.includeMetadataNote")}
              </p>
            </div>
          )}
          <div className="flex items-center justify-end gap-2 pt-4 border-t">
            <Button variant="outline" onClick={onClose} disabled={isExporting}>
              {t("usage.export.modal.cancel")}
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting && <Loader2 className="animate-spin" />}
              {isExporting
                ? t("usage.export.modal.exporting")
                : t("usage.export.modal.exportFormat", { format: exportFormat.toUpperCase() })}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EntityUsageExportModal;
