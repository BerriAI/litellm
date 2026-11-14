"use client";

import ClaudeCodePluginsPanel from "./_components/ClaudeCodePluginsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Skills() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  return <ClaudeCodePluginsPanel accessToken={accessToken} userRole={userRole} isViewOnly={isViewOnly} />;
}
