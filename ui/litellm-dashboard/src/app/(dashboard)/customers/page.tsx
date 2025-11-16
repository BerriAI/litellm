"use client";

import CustomersView from "@/components/customers_view";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const CustomersPage = () => {
  const { accessToken, userRole, userId } = useAuthorized();

  return <CustomersView accessToken={accessToken} userRole={userRole} userID={userId} />;
};

export default CustomersPage;
