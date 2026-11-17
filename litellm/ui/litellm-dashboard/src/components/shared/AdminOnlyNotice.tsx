"use client";

import React from "react";
import { useTranslation } from "react-i18next";

interface AdminOnlyNoticeProps {
  pageTitle: string;
}

export const AdminOnlyNotice: React.FC<AdminOnlyNoticeProps> = ({ pageTitle }) => {
  const { t } = useTranslation();
  return (
    <div className="p-6 w-full min-w-0 flex-1">
      <h1 className="text-2xl font-semibold text-foreground mb-2">{pageTitle}</h1>
      <p className="text-sm text-muted-foreground">{t("shell.adminOnlyNotice.description", { pageTitle })}</p>
    </div>
  );
};
