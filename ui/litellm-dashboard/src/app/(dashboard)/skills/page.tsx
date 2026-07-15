"use client";

import ClaudeCodePluginsPanel from "@/components/claude_code_plugins";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Skills() {
  const { accessToken, userRole } = useAuthorized();
  return <ClaudeCodePluginsPanel accessToken={accessToken} userRole={userRole} />;
}
