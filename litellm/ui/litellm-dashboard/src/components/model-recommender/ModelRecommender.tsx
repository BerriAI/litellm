"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ArrowLeft, ArrowRight } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Panel, Segmented } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { FOCUS, requestRecommendations, type RecommendationResponse } from "./api";
import { answersToRequest, EMPTY_ANSWERS, flowParsers, GOALS, REQUIREMENTS, type Answers } from "./flow";
import { RecommendationCard } from "./RecommendationCard";

type ChangeAnswers = (patch: Partial<Answers>) => void;

export default function ModelRecommender() {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const { accessToken } = useAuthorized();
  const [answers, setAnswers] = useQueryStates(flowParsers, { history: "push" });
  const body = useMemo(() => answersToRequest(answers), [answers]);
  const queryOptions = {
    queryKey: ["model-recommendations", accessToken, body],
    queryFn: ({ signal }: { signal: AbortSignal }) => requestRecommendations(body, accessToken ?? "", signal),
    enabled: answers.step === "results" && Boolean(accessToken),
    retry: false,
    staleTime: 30000,
  };
  const { data: result, isFetching: loading, error, refetch } = useQuery(queryOptions);
  const showResults = !loading && !error && result !== undefined;
  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>{t("title")}</PageHeaderTitle>
        <PageHeaderDescription>{t("guidedIntro")}</PageHeaderDescription>
      </PageHeader>
      {answers.step === "results" ? (
        <>
          <AnswerSummary answers={answers} onChange={setAnswers} />
          {loading && (
            <Panel title={t("finding")}>
              <Skeleton className="h-64 w-full rounded-xl" />
            </Panel>
          )}
          {error && (
            <Panel title={t("requestFailed")}>
              <p role="alert" className="mb-4 text-sm text-destructive">
                {error.message}
              </p>
              <Button variant="outline" onClick={() => void refetch()}>
                {t("retry")}
              </Button>
            </Panel>
          )}
          {showResults && <Results result={result} />}
        </>
      ) : (
        <Questions answers={answers} onChange={setAnswers} />
      )}
    </Page>
  );
}

function ExternalSwitch({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 rounded-lg border p-4 text-sm">
      <span>{t("includeExternal")}</span>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  );
}

function Questions({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const options = ["1", "2", "3"].map((step) => ({ value: step, label: t("stepNumber", { step }) }));
  const onStep = (step: string) => onChange({ step: flowParsers.step.parse(step) ?? "1" });
  const question = { "1": t("question1"), "2": t("question2"), "3": t("question3"), results: "" }[answers.step];
  const next = { "1": "2", "2": "3", "3": "results", results: "results" } as const;
  const skipAnswers: Answers = { ...EMPTY_ANSWERS, focus: answers.focus, goal: answers.goal, step: "results" };
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-sm text-muted-foreground">{t("stepOf", { step: answers.step })}</span>
        <Segmented value={answers.step} options={options} onChange={onStep} label={t("steps")} />
      </div>
      <Panel title={question} bodyClassName="gap-5">
        {answers.step === "1" && <FocusOptions answers={answers} onChange={onChange} />}
        {answers.step === "2" && <GoalOptions answers={answers} onChange={onChange} />}
        {answers.step === "3" && <MustHaves answers={answers} onChange={onChange} />}
        <div className="flex items-center justify-between gap-3 border-t pt-5">
          <Button
            variant="ghost"
            disabled={answers.step === "1"}
            onClick={() => onChange({ step: answers.step === "3" ? "2" : "1" })}
          >
            <ArrowLeft />
            {t("back")}
          </Button>
          <div className="flex gap-2">
            {answers.step === "3" && (
              <Button variant="outline" onClick={() => onChange(skipAnswers)}>
                {t("skip")}
              </Button>
            )}
            <Button onClick={() => onChange({ step: next[answers.step] })}>
              {answers.step === "3" ? t("recommend") : t("continue")}
              <ArrowRight />
            </Button>
          </div>
        </div>
      </Panel>
      <p className="text-center text-xs text-muted-foreground">{t("noInference")}</p>
    </div>
  );
}

function FocusOptions({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <>
      <p className="text-sm text-muted-foreground">{t("chooseFocus")}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {FOCUS.map((focus) => (
          <button
            key={focus}
            type="button"
            aria-pressed={answers.focus.includes(focus)}
            onClick={() =>
              onChange({
                focus: answers.focus.includes(focus)
                  ? answers.focus.filter((item) => item !== focus)
                  : [...answers.focus, focus],
              })
            }
            className="rounded-xl border bg-card p-5 text-left text-sm font-medium transition-colors hover:bg-muted/50 aria-pressed:border-[#2b3fd6] aria-pressed:bg-[#2b3fd6]/5 focus-visible:outline-2 focus-visible:outline-ring"
          >
            {t(`building.${focus}`)}
          </button>
        ))}
      </div>
    </>
  );
}

function GoalOptions({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <div role="group" aria-label={t("question2")} className="grid gap-3 sm:grid-cols-2">
      {GOALS.map((goal) => (
        <button
          key={goal}
          type="button"
          aria-pressed={answers.goal === goal}
          onClick={() => onChange({ goal })}
          className="space-y-1 rounded-xl border bg-card p-5 text-left transition-colors hover:bg-muted/50 aria-pressed:border-[#2b3fd6] aria-pressed:bg-[#2b3fd6]/5 focus-visible:outline-2 focus-visible:outline-ring"
        >
          <span className="block text-sm font-medium">{t(`goals.${goal}`)}</span>
          <span className="block text-xs text-muted-foreground">{t(`goalHints.${goal}`)}</span>
        </button>
      ))}
    </div>
  );
}

function MustHaves({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <>
      <p className="text-sm text-muted-foreground">{t("optionalRequirements")}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {REQUIREMENTS.map((requirement) => (
          <label
            key={requirement}
            className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border p-5 text-sm"
          >
            <span>{t(`mustHaves.${requirement}`)}</span>
            <Switch checked={answers[requirement]} onCheckedChange={(value) => onChange({ [requirement]: value })} />
          </label>
        ))}
      </div>
      <ExternalSwitch
        checked={answers.include_external}
        onChange={(include_external) => onChange({ include_external })}
      />
    </>
  );
}

function AnswerSummary({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const focus = (answers.focus.length ? answers.focus : (["general"] as const))
    .map((area) => t(`building.${area}`))
    .join(", ");
  const requirements = REQUIREMENTS.filter((key) => answers[key])
    .map((key) => t(`mustHaves.${key}`))
    .join(", ");
  return (
    <Panel
      title={t("yourAnswers")}
      action={
        <>
          <Button variant="outline" size="sm" onClick={() => onChange({ step: "1" })}>
            {t("editAnswers")}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onChange(EMPTY_ANSWERS)}>
            {t("startOver")}
          </Button>
        </>
      }
      bodyClassName="gap-4"
    >
      <p className="text-sm text-muted-foreground">
        {focus} · {t(`goals.${answers.goal}`)}
        {requirements && ` · ${requirements}`}
      </p>
      <ExternalSwitch
        checked={answers.include_external}
        onChange={(include_external) => onChange({ include_external })}
      />
    </Panel>
  );
}

function Results({ result }: { result: RecommendationResponse }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [first, ...rest] = result.recommendations;
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t("shortlist")}</h2>
        <p className="text-xs text-muted-foreground">
          {t("counts", {
            matching: result.candidates_considered - result.candidates_filtered,
            filtered: result.candidates_filtered,
          })}
        </p>
      </div>
      {first ? (
        <>
          <RecommendationCard model={first} position={0} />
          <div className="grid gap-5 lg:grid-cols-2">
            {rest.map((model, index) => (
              <RecommendationCard key={model.model_group} model={model} position={index + 1} />
            ))}
          </div>
        </>
      ) : (
        <Panel title={t("noMatchTitle")}>
          <p className="text-sm text-muted-foreground">{t("noMatchDescription")}</p>
        </Panel>
      )}
      <footer className="space-y-2 border-t pt-4 text-xs text-muted-foreground">
        <p>{t("relativeScores")}</p>
        <p>
          {t("sources")}{" "}
          <a href="https://epoch.ai/benchmarks" target="_blank" rel="noreferrer" className="underline">
            {t("epoch")}
          </a>{" "}
          ·{" "}
          <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer" className="underline">
            {t("license")}
          </a>
        </p>
        {result.data_sources.map((source) => (
          <span key={source.name} className="mr-4 inline-block">
            {source.name} · {source.as_of ? t("asOf", { date: source.as_of }) : t("unavailable")}
          </span>
        ))}
      </footer>
    </div>
  );
}
