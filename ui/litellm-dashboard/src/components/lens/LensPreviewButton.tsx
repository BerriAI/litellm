"use client";

import { createContext, useContext } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/button";

export const LensPreviewTarget = createContext<HTMLElement | null | undefined>(undefined);

export function LensPreviewButton({ onClick }: { onClick: () => void }) {
  const target = useContext(LensPreviewTarget);
  const button = (
    <Button variant="outline" onClick={onClick}>
      Preview sample
    </Button>
  );
  if (target === null) return null;
  return target ? createPortal(button, target) : button;
}
