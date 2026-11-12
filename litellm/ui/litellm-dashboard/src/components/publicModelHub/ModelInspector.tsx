"use client";

import { ChevronDown } from "lucide-react";
import * as React from "react";
import { Trans, useTranslation } from "react-i18next";

import { DeploymentPricingBreakdown } from "@/components/molecules/models/DeploymentPricingBreakdown";
import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import i18n from "@/i18n/i18n";

import {
  formatPerMillion,
  formatTokens,
  modeLabel,
  modelCapabilities,
  providerName,
  SECTION_TITLE,
  wildcardExample,
} from "./hubFormat";
import { Badges, CodeBlock, type Fact, Facts, PaneSection, ProviderAvatar, ProviderName } from "./HubParts";
import { curlSnippet, modelSnippet } from "./hubSnippets";
import { HealthBadge } from "./HubStatus";
import { InspectorHeader } from "./InspectorHeader";
import { productNames } from "@/i18n/untranslated/productNames";

const summary = (model: ModelGroupInfo): string | undefined => {
  if (model.description) return model.description;
  const providers = (model.providers ?? []).map(providerName).join(", ");
  return model.mode && providers
    ? i18n.t("publicModelHub.modelInspector.summary", { mode: modeLabel(model.mode), providers })
    : undefined;
};

const tokens = (value: number | undefined) =>
  value ? i18n.t("publicModelHub.modelInspector.tokens", { tokens: formatTokens(value) }) : null;

const modelFacts = (model: ModelGroupInfo): Fact[][] => {
  const providers = model.providers ?? [];
  const tags = model.custom_tags ?? [];
  return [
    [
      { label: i18n.t("publicModelHub.modelInspector.facts.mode"), value: model.mode ? modeLabel(model.mode) : null },
      { label: i18n.t("publicModelHub.modelInspector.facts.context"), value: tokens(model.max_input_tokens) },
      { label: i18n.t("publicModelHub.modelInspector.facts.maxOutput"), value: tokens(model.max_output_tokens) },
    ],
    [
      {
        label: i18n.t("publicModelHub.modelInspector.facts.inputPrice"),
        value: formatPerMillion(model.input_cost_per_token),
      },
      {
        label: i18n.t("publicModelHub.modelInspector.facts.outputPrice"),
        value: formatPerMillion(model.output_cost_per_token),
      },
    ],
    [
      { label: i18n.t("publicModelHub.modelInspector.facts.rpm"), value: model.rpm },
      { label: i18n.t("publicModelHub.modelInspector.facts.tpm"), value: model.tpm ? formatTokens(model.tpm) : null },
    ],
    [
      {
        label:
          providers.length > 1
            ? i18n.t("publicModelHub.modelInspector.facts.providers")
            : i18n.t("publicModelHub.modelInspector.facts.provider"),
        wide: true,
        value: providers.length > 0 && (
          <span className="flex flex-wrap gap-x-4 gap-y-1">
            {providers.map((provider) => (
              <ProviderName key={provider} provider={provider} />
            ))}
          </span>
        ),
      },
      {
        label: i18n.t("publicModelHub.modelInspector.facts.tags"),
        wide: true,
        value: tags.length > 0 && <Badges items={tags} />,
      },
    ],
  ];
};

function CallThisModel({ model }: { model: ModelGroupInfo }) {
  const { t } = useTranslation();
  const curl = curlSnippet(model);
  if (!curl || model.mode === "passthrough") {
    return (
      <PaneSection title={t("publicModelHub.modelInspector.callThisModel.title")}>
        <CodeBlock code={modelSnippet(model)} />
      </PaneSection>
    );
  }
  return (
    <PaneSection title={t("publicModelHub.modelInspector.callThisModel.title")}>
      <Tabs defaultValue="python" className="gap-3">
        <TabsList>
          <TabsTrigger value="python">{productNames.python}</TabsTrigger>
          <TabsTrigger value="curl">{productNames.curl}</TabsTrigger>
        </TabsList>
        <TabsContent value="python">
          <CodeBlock code={modelSnippet(model)} />
        </TabsContent>
        <TabsContent value="curl">
          <CodeBlock code={curl} />
        </TabsContent>
      </Tabs>
    </PaneSection>
  );
}

function CustomMetadata({ model }: { model: ModelGroupInfo }) {
  const { t } = useTranslation();
  const metadata = Object.entries(model.custom_metadata ?? {});
  if (metadata.length === 0) return null;
  return (
    <PaneSection title={t("publicModelHub.modelInspector.customMetadata.title")}>
      <dl className="grid grid-cols-[160px_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        {metadata.map(([key, value]) => (
          <React.Fragment key={key}>
            <dt className="text-xs font-medium text-muted-foreground">{key}</dt>
            <dd>{String(value)}</dd>
          </React.Fragment>
        ))}
      </dl>
    </PaneSection>
  );
}

function SupportedParams({ params }: { params: string[] }) {
  const { t } = useTranslation();
  if (params.length === 0) return null;
  return (
    <Collapsible>
      <PaneSection
        title={t("publicModelHub.modelInspector.supportedParams.title")}
        aside={
          <CollapsibleTrigger render={<Button variant="ghost" size="sm" />}>
            {t("publicModelHub.modelInspector.supportedParams.show", { total: params.length })}
            <ChevronDown className="text-muted-foreground" />
          </CollapsibleTrigger>
        }
      >
        <CollapsibleContent>
          <Badges items={params} mono />
        </CollapsibleContent>
      </PaneSection>
    </Collapsible>
  );
}

export function ModelInspector({ model }: { model: ModelGroupInfo }) {
  const { t } = useTranslation();
  const providers = model.providers ?? [];
  const capabilities = modelCapabilities(model);
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        media={<ProviderAvatar provider={providers[0] ?? ""} className="size-6" />}
        title={model.model_group}
        copyLabel={t("publicModelHub.modelInspector.copyName")}
        status={<HealthBadge model={model} />}
        description={summary(model)}
      />
      {model.model_group.includes("*") && (
        <Alert>
          <AlertTitle>{t("publicModelHub.modelInspector.wildcard.title")}</AlertTitle>
          <AlertDescription>
            <Trans
              i18nKey="publicModelHub.modelInspector.wildcard.description"
              components={{ example: <code className="font-mono">{wildcardExample(model.model_group)}</code> }}
            />
          </AlertDescription>
        </Alert>
      )}
      <Facts groups={modelFacts(model)} />
      <DeploymentPricingBreakdown
        deployments={model.deployment_pricing}
        heading={<h3 className={SECTION_TITLE}>{t("publicModelHub.modelInspector.pricingByDeployment")}</h3>}
      />
      {capabilities.length > 0 && (
        <PaneSection title={t("publicModelHub.modelInspector.capabilities")}>
          <Badges items={capabilities} />
        </PaneSection>
      )}
      <CallThisModel model={model} />
      <CustomMetadata model={model} />
      <SupportedParams params={model.supported_openai_params ?? []} />
    </div>
  );
}
