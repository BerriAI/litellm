import { useTerminology } from "@/contexts/TerminologyContext";
import React, { useState } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import AddGuardrailForm from "./add_guardrail_form";
import { Logo } from "@/components/molecules/logo/Logo";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";
import { GuardrailCardInfo, gardenCardDescription, gardenCardName, gardenTagLabel } from "./guardrail_garden_data";

const CATEGORY_LABEL_KEYS: Record<GuardrailCardInfo["category"], { provider: ParseKeys; type: ParseKeys }> = {
  litellm: {
    provider: "guardrails.gardenDetail.categories.litellm.provider",
    type: "guardrails.gardenDetail.categories.litellm.type",
  },
  decision_model: {
    provider: "guardrails.gardenDetail.categories.decisionModel.provider",
    type: "guardrails.gardenDetail.categories.decisionModel.type",
  },
  partner: {
    provider: "guardrails.gardenDetail.categories.partner.provider",
    type: "guardrails.gardenDetail.categories.partner.type",
  },
};

interface GuardrailDetailViewProps {
  card: GuardrailCardInfo;
  onBack: () => void;
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({ card, onBack, accessToken, onGuardrailCreated }) => {
  const { t } = useTranslation();
  const term = useTerminology();
  const name = gardenCardName(card, t);
  const description = term(gardenCardDescription(card, t));
  const [isAddFormVisible, setIsAddFormVisible] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const detailRows = [
    {
      property: t("guardrails.gardenDetail.detailRows.provider"),
      value: t(CATEGORY_LABEL_KEYS[card.category].provider),
    },
    ...(card.subcategoryKey
      ? [{ property: t("guardrails.gardenDetail.detailRows.subcategory"), value: t(card.subcategoryKey) }]
      : []),
    ...(card.category === "decision_model"
      ? [
          {
            property: t("guardrails.gardenDetail.detailRows.cost"),
            value: t("guardrails.gardenDetail.detailValues.decisionModelCost"),
          },
        ]
      : []),
    ...(card.category === "litellm"
      ? [
          {
            property: t("guardrails.gardenDetail.detailRows.cost"),
            value: t("guardrails.gardenDetail.detailValues.litellmCost"),
          },
        ]
      : []),
    ...(card.category === "litellm"
      ? [
          {
            property: t("guardrails.gardenDetail.detailRows.externalDependencies"),
            value: t("guardrails.gardenDetail.detailValues.noExternalDependencies"),
          },
        ]
      : []),
    ...(card.category === "litellm"
      ? [{ property: t("guardrails.gardenDetail.detailRows.latency"), value: card.eval?.latency || "<1ms" }]
      : []),
  ];

  const evalRows = card.eval
    ? [
        { metric: t("guardrails.gardenDetail.evalMetrics.precision"), value: `${card.eval.precision}%` },
        { metric: t("guardrails.gardenDetail.evalMetrics.recall"), value: `${card.eval.recall}%` },
        { metric: t("guardrails.gardenDetail.evalMetrics.f1Score"), value: `${card.eval.f1}%` },
        { metric: t("guardrails.gardenDetail.evalMetrics.testCases"), value: String(card.eval.testCases) },
        { metric: t("guardrails.gardenDetail.evalMetrics.falsePositives"), value: "0" },
        { metric: t("guardrails.gardenDetail.evalMetrics.falseNegatives"), value: "0" },
        { metric: t("guardrails.gardenDetail.evalMetrics.latencyP50"), value: card.eval.latency },
      ]
    : [];

  const tabs = [
    { key: "overview", label: t("guardrails.gardenDetail.tabs.overview") },
    ...(card.eval ? [{ key: "eval", label: t("guardrails.gardenDetail.tabs.evalResults") }] : []),
  ];

  return (
    <div className="mx-auto max-w-[960px]">
      {/* Back link */}
      <div
        onClick={onBack}
        className="mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground"
      >
        <ArrowLeft className="size-3" />
        <span>{name}</span>
      </div>

      {/* ── Header block (Vertex-style) ── */}
      <div className="mb-2 flex items-center gap-4">
        <Logo src={card.logo} label={name} className="w-10 h-10 rounded-lg object-contain shrink-0" />
        <h1 className="m-0 text-[28px] font-normal leading-tight text-foreground">{name}</h1>
      </div>

      <p className="m-0 mb-5 text-sm leading-relaxed text-muted-foreground">{description}</p>

      {/* Action buttons — outlined style like Vertex */}
      <div className="mb-8 flex gap-2.5">
        <Button variant="outline" className="rounded-full" onClick={() => setIsAddFormVisible(true)}>
          {t("guardrails.gardenDetail.createGuardrail")}
        </Button>
      </div>

      {/* ── Tab bar ──────────────────────────────────── */}
      <div className="mb-7 border-b border-border">
        <div className="flex">
          {tabs.map((tab) => (
            <div
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",
                activeTab === tab.key
                  ? "border-info font-medium text-info"
                  : "border-transparent font-normal text-muted-foreground",
              )}
            >
              {tab.label}
            </div>
          ))}
        </div>
      </div>

      {/* ── Tab content ──────────────────────────────── */}
      {activeTab === "overview" && (
        <div className="flex gap-16">
          {/* Left column — overview + details table */}
          <div className="min-w-0 flex-1">
            <h2 className="m-0 mb-3 text-lg font-normal text-foreground">
              {t("guardrails.gardenDetail.overviewHeading")}
            </h2>
            <p className="m-0 mb-8 text-sm leading-[1.7] text-foreground">{description}</p>

            <h2 className="m-0 mb-1 text-lg font-normal text-foreground">
              {t("guardrails.gardenDetail.detailsHeading")}
            </h2>
            <p className="m-0 mb-4 text-[13px] text-muted-foreground">{t("guardrails.gardenDetail.detailsSubtitle")}</p>

            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="w-50 py-3 text-left font-medium text-muted-foreground">
                    {t("guardrails.gardenDetail.propertyColumn")}
                  </th>
                  <th className="py-3 text-left font-medium text-muted-foreground">{name}</th>
                </tr>
              </thead>
              <tbody>
                {detailRows.map((row, i) => (
                  <tr key={i} className="border-b border-border">
                    <td className="py-3 text-foreground">{row.property}</td>
                    <td className="py-3 text-foreground">{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Right column — metadata sidebar like Vertex */}
          <div className="w-60 shrink-0">
            {/* Guardrail ID */}
            <div className="mb-7">
              <div className="mb-1 text-xs text-muted-foreground">{t("guardrails.gardenDetail.guardrailIdLabel")}</div>
              <div className="break-all text-[13px] text-foreground">litellm/{card.id}</div>
            </div>

            {/* Type */}
            <div className="mb-7">
              <div className="mb-1 text-xs text-muted-foreground">{t("guardrails.gardenDetail.typeLabel")}</div>
              <div className="text-[13px] text-foreground">{t(CATEGORY_LABEL_KEYS[card.category].type)}</div>
            </div>

            {/* Tags — pill style like Vertex */}
            {card.tags.length > 0 && (
              <div className="mb-7">
                <div className="mb-2 text-xs text-muted-foreground">{t("guardrails.gardenDetail.tagsLabel")}</div>
                <div className="flex flex-wrap gap-1.5">
                  {card.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground"
                    >
                      {gardenTagLabel(tag, t)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "eval" && (
        <div>
          <h2 className="m-0 mb-4 text-lg font-normal text-foreground">
            {t("guardrails.gardenDetail.evalResultsHeading")}
          </h2>
          <table className="w-full max-w-[560px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-border bg-muted">
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                  {t("guardrails.gardenDetail.metricColumn")}
                </th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                  {t("guardrails.gardenDetail.valueColumn")}
                </th>
              </tr>
            </thead>
            <tbody>
              {evalRows.map((row, i) => (
                <tr key={i} className="border-b border-border">
                  <td className="px-4 py-3 text-foreground">{row.metric}</td>
                  <td className="px-4 py-3 font-medium text-foreground">{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <AddGuardrailForm
        visible={isAddFormVisible}
        onClose={() => setIsAddFormVisible(false)}
        accessToken={accessToken}
        onSuccess={() => {
          setIsAddFormVisible(false);
          onGuardrailCreated();
        }}
        preset={GUARDRAIL_PRESETS[card.id]}
      />
    </div>
  );
};

export default GuardrailDetailView;
