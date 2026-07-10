"use client";

import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useAllTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import KeyInfoView from "@/components/templates/key_info_view";
import { migratedHref } from "@/utils/migratedPages";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { Button } from "antd";
import { useRouter } from "next/navigation";
import { useState } from "react";

function keyIdFromPathname(pathname: string): string {
  const segments = pathname.replace(/\/+$/, "").split("/");
  return decodeURIComponent(segments[segments.length - 1] ?? "");
}

export default function KeyDetailPage() {
  const router = useRouter();
  const { isLoading: authLoading, isAuthorized } = useAuthorized();
  const [keyId] = useState(() => (typeof window === "undefined" ? "" : keyIdFromPathname(window.location.pathname)));

  const { data: keyData, isPending } = useKeyInfo(keyId);
  const { data: teams } = useAllTeams();

  const backToKeys = () => router.push(migratedHref("api-keys"));

  if (authLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  if (!keyId || isPending) {
    return <LoadingScreen />;
  }
  if (!keyData) {
    return (
      <div className="p-4">
        <Button type="text" icon={<ArrowLeftOutlined />} onClick={backToKeys} className="mb-4">
          Back to Keys
        </Button>
        <p className="text-sm text-gray-700">Key not found</p>
      </div>
    );
  }

  return <KeyInfoView keyId={keyId} keyData={keyData} teams={teams ?? []} onClose={backToKeys} />;
}
