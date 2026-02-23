"use client";

import { MCPServers } from "@/components/mcp_tools";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const MCPServersPage = () => {
  const { accessToken, userRole, userId } = useAuthorized();

  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <MCPServers accessToken={accessToken} userRole={userRole} userID={userId} />
    </QueryClientProvider>
  );
};

export default MCPServersPage;
