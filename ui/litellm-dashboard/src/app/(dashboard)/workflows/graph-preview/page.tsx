import { notFound } from "next/navigation";

export default async function WorkflowGraphPreviewPage() {
  if (process.env.NODE_ENV === "development") {
    const { default: WorkflowGraphDemo } = await import("../../../../../tests/fixtures/workflows/WorkflowGraphDemo");
    return <WorkflowGraphDemo />;
  }
  notFound();
}
