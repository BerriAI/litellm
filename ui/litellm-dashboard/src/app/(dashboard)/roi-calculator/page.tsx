"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ROICalculatorView from "./_components/ROICalculatorView";
import ObservedROIView from "./_components/ObservedROIView";
import { useSearchParams } from "next/navigation";

export default function ROICalculatorPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const searchParams = useSearchParams();
  if (searchParams.get("prototype") === "1") return <ObservedROIView accessToken={accessToken} />;
  return <ROICalculatorView accessToken={accessToken} userRole={userRole} isViewOnly={isViewOnly} />;
}
