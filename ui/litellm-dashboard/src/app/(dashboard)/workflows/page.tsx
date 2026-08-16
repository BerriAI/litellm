"use client";

import WorkflowRuns from "./WorkflowRuns";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Workflows() {
  const { accessToken } = useAuthorized();
  return (
    <>
      <DeprecationBanner featureName="Workflows" />
      <WorkflowRuns accessToken={accessToken} />
    </>
  );
}
