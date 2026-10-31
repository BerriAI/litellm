"use client";

import { MemoryView } from "./_components/MemoryView";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { useTranslation } from "react-i18next";

export default function Memory() {
  const { t } = useTranslation();
  const { accessToken, userRole, userId } = useAuthorized();
  const canViewMemory = useCan("viewMemory");

  if (!canViewMemory) {
    return <AdminOnlyNotice pageTitle={t("memory.page.title")} />;
  }

  return <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />;
}
