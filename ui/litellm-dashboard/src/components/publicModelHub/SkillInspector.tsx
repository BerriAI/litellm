"use client";

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
  const addedAt = skill.created_at ? new Date(skill.created_at) : null;
  const added = addedAt && !Number.isNaN(addedAt.getTime()) ? addedAt.toLocaleDateString() : null;
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
            { label: t("publicModelHub.skillInspector.facts.added"), value: added },
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
