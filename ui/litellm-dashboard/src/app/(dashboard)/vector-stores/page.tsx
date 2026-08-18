"use client";

import VectorStoreManagement from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function VectorStores() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <VectorStoreManagement accessToken={accessToken} userRole={userRole} userID={userId} />;
}
