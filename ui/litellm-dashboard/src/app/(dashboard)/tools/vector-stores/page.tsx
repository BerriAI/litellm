"use client";

import VectorStoreManagement from "@/components/vector_store_management";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const VectorStoresPage = () => {
  const { accessToken, userId, userRole } = useAuthorized();

  return <VectorStoreManagement accessToken={accessToken} userID={userId} userRole={userRole} />;
};

export default VectorStoresPage;
