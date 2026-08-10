"use client";

import { Plus } from "lucide-react";
import { useMemo, useState } from "react";

import { useAutoRouters, useInvalidateAutoRouters } from "@/app/(dashboard)/hooks/models/useModels";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";
import AddAutoRouterTab from "@/components/add_model/add_auto_router_tab";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { modelDeleteCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { type ModelWriteScope } from "@/utils/modelPermissions";
import { Team } from "@/components/networking";

import { AutoRoutersTable } from "./AutoRoutersTable";
import { AutoRouterRow, toAutoRouterRows } from "./autoRouterRows";
import { useTranslation } from "react-i18next";

interface AutoRoutersPanelProps {
  accessToken: string;
  userRole: string;
  userID: string | null;
  teams: Team[] | null;
  /** Owned by the page, which knows how this caller must scope what they create. */
  createScope: ModelWriteScope;
}

export function AutoRoutersPanel({ accessToken, userRole, userID, teams, createScope }: AutoRoutersPanelProps) {
  const { t } = useTranslation("gateway");
  const canCreate = createScope !== "forbidden";
  const { data: deployments, isLoading } = useAutoRouters();
  const invalidateAutoRouters = useInvalidateAutoRouters();
  // Clicking a router opens the same ?model= drill-in the All Models table uses, so an auto
  // router gets the full ModelInfoView: Model Settings, Edit Settings, Edit Auto Router and
  // Delete. A separate detail view here would be a worse copy of it.
  const { openModel } = useModelDetailRouting();
  const [isCreating, setIsCreating] = useState(false);
  const [deletingRouter, setDeletingRouter] = useState<AutoRouterRow | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const routers = useMemo(
    () => toAutoRouterRows(deployments ?? [], { userRole, userID }, teams),
    [deployments, userRole, userID, teams],
  );

  const handleCreated = () => {
    setIsCreating(false);
    void invalidateAutoRouters();
  };

  const handleConfirmDelete = async () => {
    if (!deletingRouter) return;
    setIsDeleting(true);
    try {
      await modelDeleteCall(accessToken, deletingRouter.id);
      NotificationsManager.success(t("models.autoRouters.deleted", { name: deletingRouter.name }));
      setDeletingRouter(null);
      await invalidateAutoRouters();
    } catch (error) {
      NotificationsManager.fromBackend(t("models.autoRouters.deleteFailed", { error: String(error) }));
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="w-full space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">{t("models.autoRouters.title")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t("models.autoRouters.description")}</p>
        </div>
        {canCreate && (
          <Button onClick={() => setIsCreating(true)} className="shrink-0">
            <Plus />
            {t("models.autoRouters.add")}
          </Button>
        )}
      </div>

      <AutoRoutersTable
        routers={routers}
        isLoading={isLoading}
        canModify={canCreate}
        onRouterClick={(row) => openModel(row.id)}
        onDeleteClick={setDeletingRouter}
      />

      <Dialog open={isCreating} onOpenChange={setIsCreating}>
        {/* The form is long, so the dialog caps its height and scrolls its body rather than
            growing past the viewport. */}
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
          <DialogHeader>
            <DialogTitle>{t("models.autoRouters.add")}</DialogTitle>
            <DialogDescription>{t("models.autoRouters.dialogDescription")}</DialogDescription>
          </DialogHeader>
          <AddAutoRouterTab
            handleOk={handleCreated}
            accessToken={accessToken}
            userRole={userRole}
            userId={userID}
            createScope={createScope}
          />
        </DialogContent>
      </Dialog>

      {deletingRouter && (
        <DeleteResourceModal
          isOpen
          title={t("models.autoRouters.deleteDialog.title")}
          message={t("models.autoRouters.deleteDialog.message", { name: deletingRouter.name })}
          resourceInformationTitle={t("models.autoRouters.deleteDialog.information")}
          resourceInformation={[
            { label: t("models.autoRouters.columns.name"), value: deletingRouter.name },
            {
              label: t("models.autoRouters.columns.type"),
              value: t(
                `models.autoRouters.typeLabels.${
                  deletingRouter.typeLabel === "LLM Classifier"
                    ? "llm"
                    : deletingRouter.typeLabel === "Heuristic"
                      ? "heuristic"
                      : deletingRouter.kind
                }`,
                { defaultValue: deletingRouter.typeLabel },
              ),
            },
            { label: t("models.autoRouters.deleteDialog.id"), value: deletingRouter.id },
          ]}
          onCancel={() => setDeletingRouter(null)}
          onOk={handleConfirmDelete}
          confirmLoading={isDeleting}
        />
      )}
    </div>
  );
}
