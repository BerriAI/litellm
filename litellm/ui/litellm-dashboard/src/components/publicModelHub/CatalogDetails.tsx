"use client";

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";

export function CatalogDetails({
  name,
  loading = false,
  onClose,
  children,
}: {
  name: string | null;
  loading?: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <Dialog open={name !== null} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-h-[90dvh] gap-0 overflow-y-auto p-0 pt-8 sm:max-w-3xl" aria-describedby={undefined}>
        <DialogTitle className="sr-only">{t("publicModelHub.masterDetail.details")}</DialogTitle>
        {loading ? (
          <div className="flex flex-col gap-6 p-6" aria-busy="true">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : (
          children
        )}
      </DialogContent>
    </Dialog>
  );
}
