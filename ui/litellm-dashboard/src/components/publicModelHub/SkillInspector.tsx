"use client";

import type { Plugin } from "@/components/claude_code_plugins/types";
import { Field, FieldDescription } from "@/components/ui/field";

import { Badges, CodeBlock, Facts, PaneSection } from "./HubParts";
import { marketplaceCommand, skillInstallCommand } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

export function SkillInspector({ skill }: { skill: Plugin }) {
  const keywords = skill.keywords ?? [];
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader title={skill.name} copyLabel="Copy skill name" description={skill.description} />
      <Facts
        groups={[
          [
            { label: "Domain", value: skill.domain },
            { label: "Namespace", value: skill.namespace },
            { label: "Version", value: skill.version },
            { label: "Category", value: skill.category },
          ],
          [{ label: "Keywords", wide: true, value: keywords.length > 0 && <Badges items={keywords} /> }],
        ]}
      />
      <PaneSection title="Install in Claude Code">
        <Field>
          <FieldDescription>First time only: add this marketplace</FieldDescription>
          <CodeBlock code={marketplaceCommand()} />
        </Field>
        <Field>
          <FieldDescription>Then install the skill</FieldDescription>
          <CodeBlock code={skillInstallCommand(skill)} />
        </Field>
      </PaneSection>
    </div>
  );
}
