"use client";

import React from "react";

/**
 * Layout for the Keys section
 * This allows nested routes like /keys/{key_uuid} to inherit common structure
 */
export default function KeysLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
