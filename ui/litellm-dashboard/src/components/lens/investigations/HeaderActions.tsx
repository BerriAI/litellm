"use client";

import { useContext, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { LensPreviewTarget } from "../LensPreviewButton";

export function HeaderActions({ children }: { children: ReactNode }) {
  const target = useContext(LensPreviewTarget);
  const actions = <div className="flex flex-wrap items-center gap-3">{children}</div>;
  if (target === null) return null;
  return target ? createPortal(actions, target) : actions;
}
