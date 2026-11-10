"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { Ban, Pencil } from "lucide-react";
import { toast } from "sonner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { customerStatus, useCustomerUsage } from "./customerPortalData";
import EditCustomerDialog from "./EditCustomerDialog";
import StatusBadge from "./StatusBadge";

function UsageSummary({ spend, budget }: { spend: number; budget: EndUser["litellm_budget_table"] | undefined }) {
  const { t } = useTranslation();
  const max = budget?.max_budget ?? null;
  const tpmLimit = budget?.tpm_limit ?? null;
  const rpmLimit = budget?.rpm_limit ?? null;
  return (
    <p className="flex flex-wrap items-center gap-x-2 text-sm text-muted-foreground">
      <span>
        {t("customerPortal.detail.spendThisMonth")}{" "}
        <span className="font-medium text-foreground">{t("format.usd", { val: spend })}</span>
      </span>
      <span>
        {max === null
          ? t("customerPortal.detail.noBudget")
          : t("customerPortal.detail.ofBudget", { max: t("format.usd", { val: max }) })}
      </span>
      {tpmLimit !== null && (
        <Badge variant="outline">
          {t("customerPortal.detail.tpm", { val: t("format.number", { val: tpmLimit }) })}
        </Badge>
      )}
      {rpmLimit !== null && (
        <Badge variant="outline">
          {t("customerPortal.detail.rpm", { val: t("format.number", { val: rpmLimit }) })}
        </Badge>
      )}
    </p>
  );
}

export default function CustomerDetail({ customerId, onBack }: { customerId: string; onBack: () => void }) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const info = $api.useQuery(
    "get",
    "/customer/info",
    { params: { query: { end_user_id: customerId } } },
    { enabled: Boolean(accessToken) },
  );
  const usageQuery = useCustomerUsage(accessToken);
  const [editOpen, setEditOpen] = useState(false);
  const queryClient = useQueryClient();
  const refresh = () => {
    info.refetch();
    queryClient.invalidateQueries({ queryKey: ["get", "/customer/list"] });
  };
  const block = $api.useMutation("post", "/customer/block", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.blocked"));
      refresh();
    },
  });
  const unblock = $api.useMutation("post", "/customer/update", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.unblocked"));
      refresh();
    },
  });

  const c = info.data;
  const usage = usageQuery.data?.get(customerId);
  const spend = usage?.spend ?? 0;
  const keys = usage?.keys ?? [];
  const label = c?.alias || customerId;

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight">{label}</h1>
            {c && <StatusBadge status={customerStatus(c, spend)} />}
            <span className="font-mono text-xs text-muted-foreground">{customerId}</span>
          </div>
          <UsageSummary spend={spend} budget={c?.litellm_budget_table} />
        </div>
        <div className="flex gap-2">
          <Button variant="outline" disabled={!c} onClick={() => setEditOpen(true)}>
            <Pencil className="size-4" /> {t("customerPortal.common.edit")}
          </Button>
          {c?.blocked ? (
            <Button variant="outline" onClick={() => unblock.mutate({ body: { user_id: customerId, blocked: false } })}>
              <Ban className="size-4" /> {t("customerPortal.common.unblock")}
            </Button>
          ) : (
            <Button
              variant="outline"
              className="text-destructive"
              onClick={() => block.mutate({ body: { user_ids: [customerId] } })}
            >
              <Ban className="size-4" /> {t("customerPortal.common.block")}
            </Button>
          )}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{t("customerPortal.common.blockHint")}</p>

      <div className="rounded-lg border bg-card">
        <div className="flex items-baseline gap-3 border-b px-4 py-3">
          <h2 className="text-sm font-semibold">{t("customerPortal.detail.keysTitle")}</h2>
          <p className="text-xs text-muted-foreground">{t("customerPortal.detail.keysSubtitle")}</p>
        </div>
        {keys.length === 0 ? (
          <p className="py-10 text-center text-sm text-muted-foreground">{t("customerPortal.detail.keysEmpty")}</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">{t("customerPortal.detail.columns.key")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.detail.columns.requests")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.detail.columns.tokens")}</TableHead>
                <TableHead className="pr-4 text-right">{t("customerPortal.detail.columns.spend")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.map((k) => (
                <TableRow key={k.hash}>
                  <TableCell className="pl-4">
                    <p className="font-medium">{k.alias ?? t("customerPortal.detail.unnamedKey")}</p>
                    <p className="font-mono text-[11px] text-muted-foreground">{k.hash.slice(0, 12)}</p>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{t("format.number", { val: k.requests })}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {t("format.number", { val: k.inputTokens + k.outputTokens })}
                  </TableCell>
                  <TableCell className="pr-4 text-right tabular-nums">{t("format.usd", { val: k.spend })}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
      <Button variant="link" size="sm" className="h-auto p-0 text-xs text-muted-foreground" onClick={onBack}>
        {t("customerPortal.detail.back")}
      </Button>
      {c && <EditCustomerDialog open={editOpen} onOpenChange={setEditOpen} customer={c} onSaved={refresh} />}
    </div>
  );
}
