"use client";

import AutorouterTab from "@/app/(dashboard)/cost-optimization/_components/AutorouterTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function AutorouterPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <AutorouterTab accessToken={accessToken} userId={userId} userRole={userRole} />;
}
