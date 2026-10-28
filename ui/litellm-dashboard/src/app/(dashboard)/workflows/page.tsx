"use client";

import WorkflowRuns from "./WorkflowRuns";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";

export default function Workflows() {
  const { accessToken } = useAuthorized();
  const canViewWorkflowRuns = useCan("viewWorkflowRuns");

  if (!canViewWorkflowRuns) {
    return <AdminOnlyNotice pageTitle="Workflow Runs" />;
  }

  return (
    <>
      <DeprecationBanner featureName="Workflows" />
      <WorkflowRuns accessToken={accessToken} />
    </>
  );
}
