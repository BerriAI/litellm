"use client";

import { MemoryView } from "./components/MemoryView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Memory() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />;
}
