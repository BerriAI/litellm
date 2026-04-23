import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { createTeamAliasMap } from "@/utils/teamUtils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import React, { useMemo, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import ExportFormatSelector from "./ExportFormatSelector";
import ExportSummary from "./ExportSummary";
import ExportTypeSelector from "./ExportTypeSelector";
import type {
  EntityUsageExportModalProps,
  ExportFormat,
  ExportScope,
} from "./types";
import { handleExportCSV, handleExportJSON } from "./utils";

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
  const { data: teams, isLoading: isLoadingTeams } = useTeams();

  const entityLabel = entityType.charAt(0).toUpperCase() + entityType.slice(1);
  const modalTitle = customTitle || `Export ${entityLabel} Usage`;

  const teamAliasMap = useMemo(() => createTeamAliasMap(teams), [teams]);

  const handleExport = async (format?: ExportFormat) => {
    const formatToUse = format || exportFormat;
    setIsExporting(true);
    try {
      if (formatToUse === "csv") {
        handleExportCSV(
          spendData,
          exportScope,
          entityLabel,
          entityType,
          teamAliasMap,
        );
        NotificationsManager.success(
          `${entityLabel} usage data exported successfully as CSV`,
        );
      } else {
        handleExportJSON(
          spendData,
          exportScope,
          entityLabel,
          entityType,
          dateRange,
          selectedFilters,
          teamAliasMap,
        );
        NotificationsManager.success(
          `${entityLabel} usage data exported successfully as JSON`,
        );
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
    <Dialog
      open={isOpen}
      onOpenChange={(open) => (!open ? onClose() : undefined)}
    >
      <DialogContent className="max-w-[480px]">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            {modalTitle}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-5 py-2">
          {isLoadingTeams ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          ) : (
            <>
              <ExportSummary
                dateRange={dateRange}
                selectedFilters={selectedFilters}
              />
              <ExportTypeSelector
                value={exportScope}
                onChange={setExportScope}
                entityType={entityType}
              />
              <ExportFormatSelector
                value={exportFormat}
                onChange={setExportFormat}
              />
            </>
          )}
          {isLoadingTeams ? (
            <div className="flex items-center justify-end gap-2 pt-4 border-t">
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-24" />
            </div>
          ) : (
            <div className="flex items-center justify-end gap-2 pt-4 border-t">
              <Button
                variant="outline"
                onClick={onClose}
                disabled={isExporting}
              >
                Cancel
              </Button>
              <Button
                onClick={() => handleExport()}
                disabled={isExporting || isLoadingTeams}
              >
                {isExporting
                  ? "Exporting..."
                  : `Export ${exportFormat.toUpperCase()}`}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EntityUsageExportModal;
