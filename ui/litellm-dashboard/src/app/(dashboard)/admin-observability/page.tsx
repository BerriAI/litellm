"use client";

import { AntDLoadingSpinner } from "@/components/ui/AntDLoadingSpinner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import AdminObservabilityTable from "@/components/admin_observability/AdminObservabilityTable";

export default function AdminObservabilityPage() {
  const { accessToken } = useAuthorized();
  if (!accessToken) {
    return <AntDLoadingSpinner />;
  }
  return <AdminObservabilityTable accessToken={accessToken} />;
}
