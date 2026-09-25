"use client";

import { MCPServers } from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function McpServers() {
  const { accessToken, userRole, userId, isViewOnly } = useAuthorized();
  return <MCPServers accessToken={accessToken} userRole={userRole} userID={userId} isViewOnly={isViewOnly} />;
}
