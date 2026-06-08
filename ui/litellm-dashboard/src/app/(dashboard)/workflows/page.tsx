"use client";

import WorkflowRuns from "@/components/workflow_runs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function WorkflowsPage() {
  const { accessToken } = useAuthorized();
  return <WorkflowRuns accessToken={accessToken} />;
}
