"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ObservedROIView from "./_components/ObservedROIView";

export default function ROICalculatorPage() {
  const { accessToken, isViewOnly } = useAuthorized();
  if (!accessToken) return null;
  return <ObservedROIView key={accessToken} accessToken={accessToken} isViewOnly={isViewOnly} />;
}
