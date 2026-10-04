"use client";

import { useContext, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { LensPreviewContext } from "./LensPreviewButton";

export function HeaderActions({ children }: { children: ReactNode }) {
  const preview = useContext(LensPreviewContext);
  const actions = <div className="flex flex-wrap items-center gap-3">{children}</div>;
  if (preview === undefined) return actions;
  return preview.target ? createPortal(actions, preview.target) : null;
}
