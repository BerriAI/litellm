"use client";

import { useKeyInfo } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useAllTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import KeyInfoView from "@/components/templates/key_info_view";
import { migratedHref } from "@/utils/migratedPages";
import { useParams, useRouter } from "next/navigation";

export default function KeyDetailPage() {
  const router = useRouter();
  const params = useParams();
  const { isLoading: authLoading, isAuthorized } = useAuthorized();

  const rawKeyId = params?.keyid;
  const keyId = decodeURIComponent(Array.isArray(rawKeyId) ? rawKeyId[0] : rawKeyId ?? "");

  const { data: keyData, isPending } = useKeyInfo(keyId);
  const { data: teams } = useAllTeams();

  if (authLoading || !isAuthorized || isPending) {
    return <LoadingScreen />;
  }

  return (
    <KeyInfoView
      keyId={keyId}
      keyData={keyData ?? undefined}
      teams={teams ?? []}
      onClose={() => router.push(migratedHref("api-keys"))}
    />
  );
}
