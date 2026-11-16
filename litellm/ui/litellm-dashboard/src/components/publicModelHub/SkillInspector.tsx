"use client";

import { DateCell } from "@/components/shared/table_cells";
import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";

import { getSourceDisplayText, getSourceLink } from "@/components/claude_code_plugins/helpers";
import type { Plugin } from "@/components/claude_code_plugins/types";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription } from "@/components/ui/field";

import { Badges, CodeBlock, Facts, PaneSection } from "./HubParts";
import { marketplaceCommand, skillInstallCommand } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

function sourceValue(source: Plugin["source"]) {
  const link = getSourceLink(source);
  const href = link && source.source === "git-subdir" && source.path ? `${link}/tree/main/${source.path}` : link;
  if (!href) return getSourceDisplayText(source) || null;
  return (
    <Button
      variant="link"
      nativeButton={false}
      className="h-auto max-w-full justify-start px-0"
      render={<a href={href} target="_blank" rel="noopener noreferrer" />}
    >
      <span className="truncate">{href.replace(/^https:\/\//, "")}</span>
      <ExternalLink />
    </Button>
  );
}

export function SkillInspector({ skill }: { skill: Plugin }) {
  const { t } = useTranslation();
  const keywords = skill.keywords ?? [];
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        title={skill.name}
        copyLabel={t("publicModelHub.skillInspector.copyName")}
        description={skill.description}
      />
      <Facts
        groups={[
          [
            { label: t("publicModelHub.skillInspector.facts.domain"), value: skill.domain },
            { label: t("publicModelHub.skillInspector.facts.namespace"), value: skill.namespace },
            { label: t("publicModelHub.skillInspector.facts.version"), value: skill.version },
            { label: t("publicModelHub.skillInspector.facts.category"), value: skill.category },
          ],
          [
            { label: t("publicModelHub.skillInspector.facts.source"), wide: true, value: sourceValue(skill.source) },
            { label: t("publicModelHub.skillInspector.facts.author"), value: skill.author?.name },
            { label: t("skills.table.columns.createdAt"), value: <DateCell value={skill.created_at} /> },
            { label: t("skills.table.columns.updatedAt"), value: <DateCell value={skill.updated_at} /> },
            { label: t("skills.table.columns.createdBy"), value: skill.created_by || t("skills.table.notRecorded") },
            {
              label: t("skills.table.columns.publishedBy"),
              value: skill.published_by || t("skills.table.notRecorded"),
            },
            { label: t("skills.table.columns.approvedBy"), value: skill.approved_by || t("skills.table.notRecorded") },
            {
              label: t("skills.table.columns.securityScan"),
              value: (
                <span title={t("skills.table.scanHelp")}>
                  {t(skill.security_scan_passed ? "skills.table.publicYes" : "skills.table.publicNo")}
                </span>
              ),
            },
          ],
          [
            {
              label: t("publicModelHub.skillInspector.facts.keywords"),
              wide: true,
              value: keywords.length > 0 && <Badges items={keywords} />,
            },
          ],
        ]}
      />
      <PaneSection title={t("publicModelHub.skillInspector.install.title")}>
        <Field>
          <FieldDescription>{t("publicModelHub.skillInspector.install.marketplace")}</FieldDescription>
          <CodeBlock code={marketplaceCommand()} />
        </Field>
        <Field>
          <FieldDescription>{t("publicModelHub.skillInspector.install.skill")}</FieldDescription>
          <CodeBlock code={skillInstallCommand(skill)} />
        </Field>
      </PaneSection>
    </div>
  );
}
