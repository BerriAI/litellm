import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowLeft } from "lucide-react";
import AddGuardrailForm from "./add_guardrail_form";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";
import { GuardrailCardInfo } from "./guardrail_garden_data";

interface GuardrailDetailViewProps {
  card: GuardrailCardInfo;
  onBack: () => void;
  accessToken: string | null;
  onGuardrailCreated: () => void;
}

const GuardrailDetailView: React.FC<GuardrailDetailViewProps> = ({
  card,
  onBack,
  accessToken,
  onGuardrailCreated,
}) => {
  const [isAddFormVisible, setIsAddFormVisible] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const detailRows = [
    {
      property: "Provider",
      value:
        card.category === "litellm"
          ? "LiteLLM Content Filter"
          : "Partner Guardrail",
    },
    ...(card.subcategory
      ? [{ property: "Subcategory", value: card.subcategory }]
      : []),
    ...(card.category === "litellm"
      ? [{ property: "Cost", value: "$0 / request" }]
      : []),
    ...(card.category === "litellm"
      ? [{ property: "External Dependencies", value: "None" }]
      : []),
    ...(card.category === "litellm"
      ? [{ property: "Latency", value: card.eval?.latency || "<1ms" }]
      : []),
  ];

  const evalRows = card.eval
    ? [
        { metric: "Precision", value: `${card.eval.precision}%` },
        { metric: "Recall", value: `${card.eval.recall}%` },
        { metric: "F1 Score", value: `${card.eval.f1}%` },
        { metric: "Test Cases", value: String(card.eval.testCases) },
        { metric: "False Positives", value: "0" },
        { metric: "False Negatives", value: "0" },
        { metric: "Latency (p50)", value: card.eval.latency },
      ]
    : [];

  const tabs = [
    { key: "overview", label: "Overview" },
    ...(card.eval ? [{ key: "eval", label: "Eval Results" }] : []),
  ];

  return (
    <div className="max-w-[960px] mx-auto">
      {/* Back link */}
      <button
        type="button"
        onClick={onBack}
        className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground cursor-pointer text-sm mb-6"
      >
        <ArrowLeft className="h-3 w-3" />
        <span>{card.name}</span>
      </button>

      {/* Header block */}
      <div className="flex items-center gap-4 mb-2">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={card.logo}
          alt=""
          className="w-10 h-10 rounded-lg object-contain"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        <h1 className="text-[28px] font-normal text-foreground m-0 leading-tight">
          {card.name}
        </h1>
      </div>

      <p className="text-sm text-muted-foreground mt-0 mb-5 leading-relaxed">
        {card.description}
      </p>

      {/* Action buttons */}
      <div className="flex gap-2.5 mb-8">
        <Button
          variant="outline"
          onClick={() => setIsAddFormVisible(true)}
          className="rounded-full px-5 h-9 text-primary font-medium"
        >
          Create Guardrail
        </Button>
      </div>

      {/* Tab bar */}
      <div className="border-b border-border mb-7">
        <div className="flex gap-0">
          {tabs.map((tab) => (
            <button
              type="button"
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "px-5 py-3 text-sm cursor-pointer -mb-px border-b-[3px] border-transparent",
                activeTab === tab.key
                  ? "text-primary border-primary font-medium"
                  : "text-muted-foreground",
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      {activeTab === "overview" && (
        <div className="flex gap-16">
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-normal text-foreground mt-0 mb-3">
              Overview
            </h2>
            <p className="text-sm text-foreground/90 leading-relaxed m-0 mb-8">
              {card.description}
            </p>

            <h2 className="text-lg font-normal text-foreground mt-0 mb-1">
              Guardrail Details
            </h2>
            <p className="text-xs text-muted-foreground m-0 mb-4">
              Details are as follows
            </p>

            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 text-muted-foreground font-medium w-[200px]">
                    Property
                  </th>
                  <th className="text-left py-3 text-muted-foreground font-medium">
                    {card.name}
                  </th>
                </tr>
              </thead>
              <tbody>
                {detailRows.map((row, i) => (
                  <tr key={i} className="border-b border-border/60">
                    <td className="py-3 text-foreground/90">{row.property}</td>
                    <td className="py-3 text-foreground">{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Right column — metadata sidebar */}
          <div className="w-60 flex-shrink-0">
            <div className="mb-7">
              <div className="text-xs text-muted-foreground mb-1">
                Guardrail ID
              </div>
              <div className="text-sm text-foreground break-all">
                litellm/{card.id}
              </div>
            </div>

            <div className="mb-7">
              <div className="text-xs text-muted-foreground mb-1">Type</div>
              <div className="text-sm text-foreground">
                {card.category === "litellm" ? "Content Filter" : "Partner"}
              </div>
            </div>

            {card.tags.length > 0 && (
              <div className="mb-7">
                <div className="text-xs text-muted-foreground mb-2">Tags</div>
                <div className="flex flex-wrap gap-1.5">
                  {card.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-3 py-1 rounded-2xl border border-border text-foreground/90 bg-background"
                    >
                      {tag}
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
          <h2 className="text-lg font-normal text-foreground mt-0 mb-4">
            Eval Results
          </h2>
          <table className="w-full max-w-[560px] border-collapse text-sm">
            <thead>
              <tr className="bg-muted border-b border-border">
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">
                  Metric
                </th>
                <th className="text-left px-4 py-3 text-muted-foreground font-medium">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {evalRows.map((row, i) => (
                <tr key={i} className="border-b border-border/60">
                  <td className="px-4 py-3 text-foreground/90">{row.metric}</td>
                  <td className="px-4 py-3 text-foreground font-medium">
                    {row.value}
                  </td>
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
