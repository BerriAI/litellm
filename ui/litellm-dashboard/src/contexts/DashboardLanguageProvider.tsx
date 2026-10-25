"use client";

import type { ReactNode } from "react";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { getProxyBaseUrl } from "@/components/networking";
import { useAuth } from "./AuthContext";
import { LanguageProvider } from "./LanguageContext";

export default function DashboardLanguageProvider({ children }: { children: ReactNode }) {
  const { userID, authLoading } = useAuth();
  // 运行时代理地址就绪后才读取配置，登录页和业务页共用同一语言上下文
  const { data } = useUISettings({ staleTime: 60_000, refetchInterval: 60_000, enabled: !authLoading });
  return (
    <LanguageProvider
      userId={userID}
      defaultLanguage={data?.values?.default_ui_language}
      storageScope={getProxyBaseUrl() ?? ""}
    >
      {children}
    </LanguageProvider>
  );
}
