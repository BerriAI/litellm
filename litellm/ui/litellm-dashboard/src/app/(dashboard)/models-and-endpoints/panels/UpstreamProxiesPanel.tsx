"use client";

import { SortingState } from "@tanstack/react-table";
import { Network, Plus } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useUpstreamProxies } from "@/app/(dashboard)/hooks/upstreamProxies/useUpstreamProxies";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import {
  upstreamProxyAddress,
  upstreamProxyCreateCall,
  upstreamProxyDeleteCall,
  upstreamProxyUpdateCall,
  type CreateUpstreamProxy,
  type UpdateUpstreamProxy,
  type UpstreamProxy,
} from "@/components/upstream_proxies/api";
import { getUpstreamProxyColumns } from "@/components/upstream_proxies/UpstreamProxyColumns";
import UpstreamProxyModal from "@/components/upstream_proxies/UpstreamProxyModal";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";

const DEFAULT_SORTING: SortingState = [{ id: "name", desc: false }];

type Dialog =
  | { kind: "closed" }
  | { kind: "add" }
  | { kind: "edit"; proxy: UpstreamProxy }
  | { kind: "delete"; proxy: UpstreamProxy };

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Network className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("upstreamProxies.panel.emptyTitle")}</div>
      <div className="text-sm text-muted-foreground">{t("upstreamProxies.panel.emptyDescription")}</div>
    </div>
  );
}

export default function UpstreamProxiesPanel() {
  const { t } = useTranslation();
  const { accessToken, userRole } = useAuthorized();
  const canModify = isProxyAdminRole(userRole ?? "");
  const { data, isLoading, refetch } = useUpstreamProxies();
  const [dialog, setDialog] = useState<Dialog>({ kind: "closed" });
  const [isDeleting, setIsDeleting] = useState(false);
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const close = () => setDialog({ kind: "closed" });

  const columnDeps = {
    t,
    canModify,
    onEdit: (proxy: UpstreamProxy) => setDialog({ kind: "edit", proxy }),
    onDelete: (proxy: UpstreamProxy) => setDialog({ kind: "delete", proxy }),
  };
  const columns = getUpstreamProxyColumns(columnDeps);

  const run = async (action: () => Promise<unknown>, success: string): Promise<void> => {
    if (!accessToken) {
      return;
    }
    try {
      await action();
      toast.success(success);
      close();
      await refetch();
    } catch (error) {
      toast.error(extractProxyErrorMessage(error));
    }
  };

  const create = (proxy: CreateUpstreamProxy) =>
    run(() => upstreamProxyCreateCall(accessToken!, proxy), t("upstreamProxies.panel.added", { name: proxy.name }));

  const update = (name: string, change: UpdateUpstreamProxy) =>
    run(() => upstreamProxyUpdateCall(accessToken!, name, change), t("upstreamProxies.panel.updated", { name }));

  const remove = async (proxy: UpstreamProxy) => {
    setIsDeleting(true);
    await run(
      () => upstreamProxyDeleteCall(accessToken!, proxy.name),
      t("upstreamProxies.panel.deleted", { name: proxy.name }),
    );
    setIsDeleting(false);
  };

  return (
    <div className="mx-auto flex w-full flex-auto flex-col gap-4 overflow-y-auto p-2">
      <div className="flex items-center justify-between gap-4">
        <p className="text-sm text-muted-foreground">{t("upstreamProxies.panel.description")}</p>
        {canModify && (
          <Button onClick={() => setDialog({ kind: "add" })}>
            <Plus className="size-4" />
            {t("upstreamProxies.panel.addProxy")}
          </Button>
        )}
      </div>

      <DataTable
        data={data?.proxies ?? []}
        paginationMode="client"
        columns={columns}
        getRowId={(proxy) => proxy.name}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={setSorting}
        isLoading={isLoading}
        loadingMessage={t("upstreamProxies.panel.loading")}
        noDataMessage={<EmptyState />}
        size="compact"
      />

      {dialog.kind === "add" && <UpstreamProxyModal mode="add" open onCancel={close} onSubmit={create} />}
      {dialog.kind === "edit" && (
        <UpstreamProxyModal mode="edit" open proxy={dialog.proxy} onCancel={close} onSubmit={update} />
      )}
      <DeleteResourceModal
        isOpen={dialog.kind === "delete"}
        onCancel={close}
        onOk={() => dialog.kind === "delete" && void remove(dialog.proxy)}
        title={t("upstreamProxies.panel.deleteTitle")}
        message={t("upstreamProxies.panel.deleteMessage")}
        resourceInformationTitle={t("upstreamProxies.panel.deleteInfoTitle")}
        resourceInformation={
          dialog.kind === "delete"
            ? [
                { label: t("upstreamProxies.columns.name"), value: dialog.proxy.name },
                { label: t("upstreamProxies.columns.address"), value: upstreamProxyAddress(dialog.proxy), code: true },
              ]
            : []
        }
        confirmLoading={isDeleting}
        requiredConfirmation={dialog.kind === "delete" ? dialog.proxy.name : undefined}
      />
    </div>
  );
}
