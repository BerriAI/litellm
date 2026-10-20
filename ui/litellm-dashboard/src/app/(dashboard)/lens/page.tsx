"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { EmbeddedLens } from "@/components/lens/EmbeddedLens";

export default function LensPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  if (!accessToken) return null;
  return <EmbeddedLens accessToken={accessToken} userRole={userRole ?? ""} readOnly={isViewOnly} />;
}
