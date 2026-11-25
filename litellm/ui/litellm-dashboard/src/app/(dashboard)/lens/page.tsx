"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { EmbeddedLens } from "@/components/lens/EmbeddedLens";
import { useAuth } from "@/contexts/AuthContext";

export default function LensPage() {
  const { accessToken } = useAuthorized();
  const { token } = useAuth();
  if (!accessToken) return null;
  return <EmbeddedLens dashboardToken={token} />;
}
