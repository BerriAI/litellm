"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ROICalculatorView from "./_components/ROICalculatorView";

export default function ROICalculatorPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  return <ROICalculatorView accessToken={accessToken} userRole={userRole} isViewOnly={isViewOnly} />;
}
