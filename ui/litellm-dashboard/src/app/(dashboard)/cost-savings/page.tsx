"use client";

import CostSavingsView from "./CostSavingsView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostSavingsPage() {
  useAuthorized();
  return <CostSavingsView />;
}
