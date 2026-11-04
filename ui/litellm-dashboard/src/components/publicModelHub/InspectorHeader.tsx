"use client";

import { X } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { SheetClose } from "@/components/ui/sheet";
import { cn } from "@/lib/cva.config";

import { SECTION_TITLE } from "./hubFormat";
import { CopyIconButton } from "./HubParts";

export const InSheet = React.createContext(false);

export function InspectorHeader({
  media,
  title,
  copyLabel,
  status,
  description,
}: {
  media?: React.ReactNode;
  title: string;
  copyLabel: string;
  status?: React.ReactNode;
  description?: React.ReactNode;
}) {
  const inSheet = React.useContext(InSheet);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex min-w-0 items-center gap-2">
        {media}
        <h2 className={cn("min-w-0 wrap-break-word", SECTION_TITLE)}>{title}</h2>
        <CopyIconButton value={title} label={copyLabel} className="size-7" />
        <span className="flex-1" />
        {status && <span className="hidden sm:flex">{status}</span>}
        {inSheet && (
          <SheetClose
            render={<Button variant="ghost" size="icon-sm" aria-label="Close" className="text-muted-foreground" />}
          >
            <X />
          </SheetClose>
        )}
      </div>
      {status && <span className="flex sm:hidden">{status}</span>}
      {description && <p className="text-sm text-muted-foreground">{description}</p>}
    </div>
  );
}
