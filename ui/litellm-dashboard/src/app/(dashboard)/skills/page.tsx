"use client";

import ClaudeCodePluginsPanel from "./_components/ClaudeCodePluginsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Skills() {
  const { accessToken, userRole } = useAuthorized();
  return <ClaudeCodePluginsPanel accessToken={accessToken} userRole={userRole} />;
}
