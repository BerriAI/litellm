"use client";

import type { ReactNode } from "react";
import { Brain, Image, Wrench, ScanText, Check, Globe } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Logo } from "@/components/molecules/logo/Logo";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { FOCUS, PRIORITIES } from "./api";
import { REQUIREMENTS, STEPS, toggleValue, weightMix, type Answers } from "./flow";

export type ChangeAnswers = (patch: Partial<Answers>) => void;
const COLORS = {
  intelligence: "bg-[#2b3fd6]",
  speed: "bg-amber-500 dark:bg-amber-400",
  cost: "bg-emerald-600 dark:bg-emerald-400",
};
const ICONS = { reasoning: Brain, vision: Image, function_calling: Wrench, long_context: ScanText };

export function Progress({ step }: { step: Answers["step"] }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div className="flex items-center gap-4 py-2" aria-label={t("steps")}>
      <span className="text-sm text-muted-foreground">{t("stepOf", { step })}</span>
      <div className="flex gap-1.5" aria-hidden="true">
        {STEPS.map((item) => (
          <span
            key={item}
            className={`h-1.5 w-9 rounded-sm ${Number(item) <= Number(step) ? "bg-[#2b3fd6]" : "bg-muted"}`}
          />
        ))}
      </div>
    </div>
  );
}

export function PrioritySelectors({ answers, onChange }: { answers: Answers; onChange?: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div className="space-y-8">
      <div className="grid gap-4 md:grid-cols-3">
        {PRIORITIES.map((kind) => (
          <div key={kind} className="space-y-5 rounded-xl border bg-card p-5">
            <div className="space-y-2">
              <h3 className="flex items-center gap-2 text-sm font-semibold">
                <span className={`size-2.5 rounded-sm ${COLORS[kind]}`} />
                {t(`priorities.${kind}`)}
              </h3>
              {onChange && <p className="min-h-8 text-xs text-muted-foreground">{t(`signalHints.${kind}`)}</p>}
            </div>
            <div className="space-y-2">
              <p className="text-right text-xs font-medium">{t(`importance.${answers[kind] as 0 | 1 | 2 | 3}`)}</p>
              <div className="relative h-8">
                <div className="pointer-events-none absolute inset-x-2 top-3.5 h-1 rounded-full bg-muted">
                  <div
                    className={`h-full rounded-full ${COLORS[kind]}`}
                    style={{ width: `${(answers[kind] / 3) * 100}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-between">
                    {[0, 1, 2, 3].map((stop) => (
                      <span
                        key={stop}
                        className={`size-2 rounded-full ${stop <= answers[kind] ? COLORS[kind] : "bg-border"}`}
                      />
                    ))}
                  </div>
                </div>
                {onChange && (
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={1}
                    value={answers[kind]}
                    aria-label={t(`priorities.${kind}`)}
                    aria-valuetext={t(`importance.${answers[kind] as 0 | 1 | 2 | 3}`)}
                    onChange={(event) => onChange({ [kind]: Number(event.target.value) })}
                    className="relative h-8 w-full cursor-pointer appearance-none rounded bg-transparent focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:bg-foreground [&::-moz-range-thumb]:size-3 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-foreground"
                  />
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      {onChange && (
        <div className="mx-auto max-w-lg space-y-3">
          <div className="flex h-2 overflow-hidden rounded-full bg-muted" role="img" aria-label={t("weightMix")}>
            {weightMix(answers).map(({ key, percent }) => (
              <span key={key} className={COLORS[key]} style={{ width: `${percent}%` }} />
            ))}
          </div>
          <div className="flex flex-wrap justify-center gap-4 text-xs text-muted-foreground">
            {weightMix(answers).map(({ key, percent }) => (
              <span key={key} className="flex items-center gap-1.5">
                <span className={`size-2 rounded-sm ${COLORS[key]}`} />
                {t(`priorities.${key}`)} {Math.round(percent)}%
              </span>
            ))}
          </div>
          {PRIORITIES.every((key) => answers[key] === 0) && (
            <p className="text-center text-xs text-muted-foreground">{t("neutralWeights")}</p>
          )}
        </div>
      )}
    </div>
  );
}

function ChoiceTile({ selected, onClick, children }: { selected: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      className="flex items-center gap-3 rounded-xl border bg-card p-5 text-left transition-colors hover:bg-muted/50 aria-pressed:border-[#2b3fd6] aria-pressed:bg-[#2b3fd6]/5 focus-visible:outline-2 focus-visible:outline-ring"
    >
      <span className="flex min-w-0 flex-1 items-center gap-3 text-sm">{children}</span>
      <span
        aria-hidden="true"
        className={`flex size-5 shrink-0 items-center justify-center rounded-full border ${selected ? "border-[#2b3fd6] bg-[#2b3fd6] text-white" : "border-border"}`}
      >
        {selected && <Check className="size-3" />}
      </span>
    </button>
  );
}

export function RequirementsSelectors({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {REQUIREMENTS.map((key) => {
        const Icon = ICONS[key];
        return (
          <ChoiceTile key={key} selected={answers[key]} onClick={() => onChange({ [key]: !answers[key] })}>
            <Icon className="size-4 shrink-0 text-muted-foreground" />
            {t(`mustHaves.${key}`)}
          </ChoiceTile>
        );
      })}
      <label className="flex cursor-pointer items-center gap-3 rounded-xl border bg-card p-5 text-sm sm:col-span-2">
        <Globe className="size-4 shrink-0 text-muted-foreground" />
        <span className="flex-1">{t("includeExternal")}</span>
        <Switch
          checked={answers.include_external}
          onCheckedChange={(include_external) => onChange({ include_external })}
        />
      </label>
    </div>
  );
}

export function FocusSelectors({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {FOCUS.map((focus) => (
        <ChoiceTile
          key={focus}
          selected={answers.focus.includes(focus)}
          onClick={() => onChange({ focus: toggleValue(answers.focus, focus) })}
        >
          <span>
            <span className="block font-medium">{t(`building.${focus}`)}</span>
            <span className="mt-1 block text-xs text-muted-foreground">{t(`benchmarkHints.${focus}`)}</span>
          </span>
        </ChoiceTile>
      ))}
    </div>
  );
}

export function ProviderSelectors({
  answers,
  onChange,
  providers,
}: {
  answers: Answers;
  onChange: ChangeAnswers;
  providers: readonly string[];
}) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">{t("providerHint")}</p>
        <Button variant="outline" size="sm" onClick={() => onChange({ providers: [...providers] })}>
          {t("selectAll")}
        </Button>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {providers.map((provider) => (
          <ChoiceTile
            key={provider}
            selected={answers.providers.includes(provider)}
            onClick={() => onChange({ providers: toggleValue(answers.providers, provider) })}
          >
            <Logo provider={provider} className="size-5 shrink-0" />
            {getProviderLogoAndName(provider).displayName}
          </ChoiceTile>
        ))}
      </div>
      {providers.length === 0 && <p className="text-sm text-muted-foreground">{t("noProviders")}</p>}
    </div>
  );
}
