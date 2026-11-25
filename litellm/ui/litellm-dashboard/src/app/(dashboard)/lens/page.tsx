"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { EmbeddedLens } from "@/components/lens/EmbeddedLens";
import { getGlobalLitellmHeaderName } from "@/components/networking";

export default function LensPage() {
  const { accessToken, isAuthorized } = useAuthorized();
  if (!isAuthorized || !accessToken) return null;
  return <EmbeddedLens accessToken={accessToken} headerName={getGlobalLitellmHeaderName()} />;
}
