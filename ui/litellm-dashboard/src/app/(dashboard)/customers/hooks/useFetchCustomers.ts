import { useCallback, useState } from "react";
import { allEndUsersCall } from "@/components/networking";

interface UseFetchCustomersProps {
  setCustomers: (customers: any[]) => void;
}

const useFetchCustomers = ({ setCustomers }: UseFetchCustomersProps) => {
  const [lastRefreshed, setLastRefreshed] = useState(
    new Date().toLocaleString("en-US")
  );

  const onRefreshClick = useCallback(
    async (accessToken: string | null) => {
      if (!accessToken) return;

      try {
        const data = await allEndUsersCall(accessToken);
        if (data) {
          setCustomers(data);
        }
        setLastRefreshed(new Date().toLocaleString("en-US"));
      } catch (error) {
        console.error("Error fetching customers:", error);
      }
    },
    [setCustomers]
  );

  return {
    lastRefreshed,
    onRefreshClick,
  };
};

export default useFetchCustomers;
