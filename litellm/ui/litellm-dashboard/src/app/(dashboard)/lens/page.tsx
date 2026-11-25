"use client";

import { EmbeddedLens } from "@/components/lens/EmbeddedLens";
import { useAuth } from "@/contexts/AuthContext";
import { checkTokenValidity, decodeToken } from "@/utils/jwtUtils";

export default function LensPage() {
  const { token, authLoading } = useAuth();
  if (authLoading || !checkTokenValidity(token)) return null;
  const session: Record<string, unknown> | null = decodeToken(token);
  const accessToken = session?.key;
  if (typeof accessToken !== "string" || !accessToken.trim()) return null;
  const configuredHeader = session?.auth_header_name;
  const headerName =
    typeof configuredHeader === "string" && configuredHeader.trim() ? configuredHeader : "Authorization";
  return <EmbeddedLens accessToken={accessToken} headerName={headerName} />;
}
