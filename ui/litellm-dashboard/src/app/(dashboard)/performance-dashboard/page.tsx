"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import PerformanceDashboardView from "@/components/PerformanceDashboard/PerformanceDashboardView";

const PerformanceDashboardPage = () => {
  useAuthorized();
  const queryClient = new QueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <PerformanceDashboardView />
    </QueryClientProvider>
  );
};

export default PerformanceDashboardPage;
