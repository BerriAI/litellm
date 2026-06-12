"use client";

import VectorStoreManagement from "@/components/vector_store_management";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function VectorStores() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <VectorStoreManagement accessToken={accessToken} userRole={userRole} userID={userId} />;
}
