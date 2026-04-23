"use client";

import React from "react";

/**
 * Legacy wrapper kept for backwards-compatibility during the phase-1
 * shadcn migration. Previously registered antd notification/message
 * instances with the global managers. The managers now delegate to
 * sonner (rendered globally via `<Toaster />` in the root layout), so
 * this component is a passthrough.
 *
 * Will be deleted entirely in the "drop AntdGlobalProvider" cleanup
 * task after antd has been uninstalled.
 */
export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
