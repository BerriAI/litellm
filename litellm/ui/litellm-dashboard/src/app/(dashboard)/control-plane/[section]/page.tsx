import { ControlPlanePage } from "@/components/ControlPlanePage";
import { controlPlanePages } from "@/lib/controlPlaneNavigation";

export function generateStaticParams() {
  return Object.keys(controlPlanePages).map((section) => ({ section }));
}

export default async function Page({ params }: { params: Promise<{ section: string }> }) {
  const { section } = await params;
  return <ControlPlanePage section={section} />;
}
