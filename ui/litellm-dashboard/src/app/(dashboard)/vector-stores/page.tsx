"use client";

import VectorStoreManagement from "@/components/vector_store_management";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function VectorStoresRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <VectorStoreManagement accessToken={accessToken} userRole={userRole} userID={userId} />;
}
