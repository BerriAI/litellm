"use client";

import WorkflowRuns from "./WorkflowRuns";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Workflows() {
  const { accessToken } = useAuthorized();
  return <WorkflowRuns accessToken={accessToken} />;
}
