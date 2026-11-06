import { notFound } from "next/navigation";

export default async function WorkflowRunPreviewPage() {
  if (process.env.NODE_ENV !== "development") notFound();
  const { default: WorkflowRunDemo } = await import("../../../../../tests/fixtures/workflow-runs/WorkflowRunDemo");
  return <WorkflowRunDemo />;
}
