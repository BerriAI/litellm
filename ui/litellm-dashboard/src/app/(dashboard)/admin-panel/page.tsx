"use client";

import AdminPanel from "@/components/AdminPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

export default function AdminPanelPage() {
  const { accessToken } = useAuthorized();
  const proxySettings = useProxySettings(accessToken);
  return <AdminPanel proxySettings={proxySettings} />;
}
