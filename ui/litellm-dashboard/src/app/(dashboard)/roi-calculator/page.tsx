"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ROICalculatorView from "./_components/ROICalculatorView";

export default function ROICalculatorPage() {
  const { accessToken } = useAuthorized();
  return <ROICalculatorView accessToken={accessToken} />;
}
