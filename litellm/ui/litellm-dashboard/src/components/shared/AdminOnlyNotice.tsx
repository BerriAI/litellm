"use client";

import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/cva.config";

interface AdminOnlyNoticeProps {
  pageTitle: string;
  framed?: boolean;
}

export const AdminOnlyNotice: React.FC<AdminOnlyNoticeProps> = ({ pageTitle, framed = false }) => {
  const { t } = useTranslation();
  return (
    <div className={cn(!framed && "p-6", "w-full min-w-0 flex-1")}>
      <h1 className="text-2xl font-semibold text-foreground mb-2">{pageTitle}</h1>
      <p className="text-sm text-muted-foreground">{t("shell.adminOnlyNotice.description", { pageTitle })}</p>
    </div>
  );
};
