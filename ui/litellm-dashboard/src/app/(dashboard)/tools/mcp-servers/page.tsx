"use client";

import { MCPServers } from "@/components/mcp_tools";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const MCPServersPage = () => {
  const { accessToken, userRole, userId } = useAuthorized();

  return <MCPServers accessToken={accessToken} userRole={userRole} userID={userId} />;
};

export default MCPServersPage;
