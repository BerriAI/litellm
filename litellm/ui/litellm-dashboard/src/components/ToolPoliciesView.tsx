"use client";

import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import { ShieldCheck } from "lucide-react";

import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { ToolDetail } from "@/components/ToolDetail";
import { ToolPoliciesPanel } from "@/components/ToolPolicies/ToolPoliciesPanel";

type View = { type: "overview" } | { type: "detail"; toolName: string };

interface ToolPoliciesViewProps {
  accessToken: string | null;
}

export default function ToolPoliciesView({ accessToken }: ToolPoliciesViewProps) {
  const { t } = useTranslation();
  const canViewToolPolicies = useCan("viewToolPolicies");
  const [view, setView] = useState<View>({ type: "overview" });

  const handleSelectTool = (toolName: string) => {
    setView({ type: "detail", toolName });
  };

  const handleBack = () => {
    setView({ type: "overview" });
  };

  if (!canViewToolPolicies) {
    return (
      <Page>
        <PageHeader>
          <PageHeaderTitle>
            <ShieldCheck />
            {t("toolPolicies.view.adminOnly.title")}
          </PageHeaderTitle>
          <PageHeaderDescription>{t("toolPolicies.view.adminOnly.message")}</PageHeaderDescription>
        </PageHeader>
      </Page>
    );
  }

  return (
    <Page>
      {view.type === "detail" ? (
        <ToolDetail toolName={view.toolName} onBack={handleBack} accessToken={accessToken} />
      ) : (
        <ToolPoliciesPanel accessToken={accessToken} onSelectTool={handleSelectTool} />
      )}
    </Page>
  );
}
