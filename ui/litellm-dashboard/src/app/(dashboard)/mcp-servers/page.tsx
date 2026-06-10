"use client";

import { MCPServers } from "@/components/mcp_tools";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function McpServersRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <MCPServers accessToken={accessToken} userRole={userRole} userID={userId} />;
}
