"use client";

import CustomersView from "@/app/(dashboard)/customers/CustomersView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useEffect, useState } from "react";
import { allEndUsersCall, type Customer } from "@/components/networking";

const CustomersPage = () => {
  const { accessToken, userId, userRole } = useAuthorized();
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchCustomers = async () => {
      if (!accessToken) return;

      setIsLoading(true);
      try {
        const response = await allEndUsersCall(accessToken);
        if (response) {
          setCustomers(Array.isArray(response) ? response : []);
        }
      } catch (error) {
        console.error("Error fetching customers:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCustomers();
  }, [accessToken]);

  return (
    <CustomersView
      customers={customers}
      setCustomers={setCustomers}
      accessToken={accessToken}
      userID={userId}
      userRole={userRole}
      isLoading={isLoading}
    />
  );
};

export default CustomersPage;
