"use client";

import { MemoryView } from "@/components/MemoryView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function MemoryRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />;
}
