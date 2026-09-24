"use client";

import * as React from "react";

import { ToolbarSeparator } from "./ToolbarSeparator";

interface EmbeddedTabsSlots {
  leadingControls: React.ReactNode;
  utilities: React.ReactNode;
}

interface PageHeaderProps {
  title: React.ReactNode;
  subtitle: React.ReactNode;
  icon: React.ReactNode;
  primaryAction?: React.ReactNode;
  tabs?: React.ReactNode | ((slots: EmbeddedTabsSlots) => React.ReactNode);
  utilities?: React.ReactNode;
}

export function PageHeader({ title, subtitle, icon, primaryAction, tabs, utilities }: PageHeaderProps) {
  const leadingControls =
    primaryAction == null ? null : (
      <div className="flex h-9 items-center">
        {primaryAction}
        {tabs != null && <ToolbarSeparator className="mx-4 h-6" />}
      </div>
    );
  const utilityControls = utilities == null ? null : <div className="flex items-center gap-2">{utilities}</div>;
  const hasControlRow = primaryAction != null || tabs != null || utilities != null;

  return (
    <div>
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className="flex size-5 flex-none items-center justify-center text-foreground [&_svg]:size-5 [&_svg]:stroke-[1.75]"
        >
          {icon}
        </span>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">{title}</h1>
      </div>
      <p className="mt-1.5 text-sm text-muted-foreground">{subtitle}</p>

      {typeof tabs === "function" ? (
        <div className="mt-5">{tabs({ leadingControls, utilities: utilityControls })}</div>
      ) : (
        hasControlRow && (
          <div className="mt-5 flex h-9 items-center" role="group" aria-label="Page controls">
            {leadingControls}
            {tabs}
            {utilityControls != null && <div className="ml-auto">{utilityControls}</div>}
          </div>
        )
      )}
    </div>
  );
}
