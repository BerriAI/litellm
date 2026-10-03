"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { LensWorkspace } from "./_components/LensWorkspace";

export default function LensPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  if (!accessToken) return null;
  return <LensWorkspace accessToken={accessToken} userRole={userRole ?? ""} readOnly={isViewOnly} />;
}
