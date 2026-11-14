"use client";

import React from "react";
import { useTranslation } from "react-i18next";

import { MemoryRow } from "@/components/networking";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";

interface MemoryDetailDrawerProps {
  row: MemoryRow | null;
  onClose: () => void;
}

const CODE_CLASS = "rounded-sm border border-border bg-muted px-1 py-0.5 font-mono text-xs text-foreground";
const BLOCK_CLASS = "mt-1 rounded-md bg-muted p-3 font-mono whitespace-pre-wrap text-foreground";
const LABEL_CLASS = "text-sm font-semibold text-foreground";

function formatTimestamp(ts?: string): string {
  if (!ts) return "—";
  try {
    const d = new Date(ts);
    return d.toLocaleString();
  } catch {
    return ts;
  }
}

export function MemoryDetailDrawer({ row, onClose }: MemoryDetailDrawerProps) {
  const { t } = useTranslation();
  return (
    <Sheet
      open={!!row}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:w-full data-[side=right]:max-w-full data-[side=right]:sm:w-[720px] data-[side=right]:sm:max-w-full">
        <SheetHeader className="border-b">
          <SheetTitle>
            {row ? <code className={CODE_CLASS}>{row.key}</code> : t("memory.detailDrawer.title")}
          </SheetTitle>
        </SheetHeader>
        {row && (
          <div className="flex flex-col gap-4 px-4 pb-4">
            <div className="flex flex-wrap gap-x-8 gap-y-3">
              <div>
                <span className={`block ${LABEL_CLASS}`}>{t("memory.detailDrawer.memoryId")}</span>
                <code className={CODE_CLASS}>{row.memory_id}</code>
              </div>
              <div>
                <span className={`block ${LABEL_CLASS}`}>{t("memory.detailDrawer.userId")}</span>
                <span className={row.user_id ? "text-sm text-foreground" : "text-sm text-muted-foreground"}>
                  {row.user_id ?? "-"}
                </span>
              </div>
              <div>
                <span className={`block ${LABEL_CLASS}`}>{t("memory.detailDrawer.teamId")}</span>
                <span className={row.team_id ? "text-sm text-foreground" : "text-sm text-muted-foreground"}>
                  {row.team_id ?? "-"}
                </span>
              </div>
            </div>
            <div>
              <span className={LABEL_CLASS}>{t("memory.detailDrawer.value")}</span>
              <p className={`${BLOCK_CLASS} text-[13px]`}>{row.value}</p>
            </div>
            {row.metadata !== undefined && row.metadata !== null && (
              <div>
                <span className={LABEL_CLASS}>{t("memory.detailDrawer.metadata")}</span>
                <p className={`${BLOCK_CLASS} text-xs`}>{JSON.stringify(row.metadata, null, 2)}</p>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span>
                {t("memory.detailDrawer.created", { time: formatTimestamp(row.created_at) })}
                {row.created_by ? t("memory.detailDrawer.by", { name: row.created_by }) : ""}
              </span>
              <span aria-hidden="true">·</span>
              <span>
                {t("memory.detailDrawer.updated", { time: formatTimestamp(row.updated_at) })}
                {row.updated_by ? t("memory.detailDrawer.by", { name: row.updated_by }) : ""}
              </span>
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}

export default MemoryDetailDrawer;
