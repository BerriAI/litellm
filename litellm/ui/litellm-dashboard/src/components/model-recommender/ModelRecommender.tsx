"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ArrowLeft, ArrowRight, Share2 } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Panel } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { requestRecommendations, type RecommendationResponse } from "./api";
import { answersToRequest, EMPTY_ANSWERS, flowParsers, STEPS, type Answers } from "./flow";
import { RecommendationCard } from "./RecommendationCard";
import { MoreOptions } from "./MoreOptions";
import {
  Progress,
  PrioritySelectors,
  RequirementsSelectors,
  FocusSelectors,
  ProviderSelectors,
  type ChangeAnswers,
} from "./Selectors";

export default function ModelRecommender() {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const { accessToken } = useAuthorized();
  const [answers, setAnswers] = useQueryStates(flowParsers, { history: "push" });
  const body = useMemo(() => answersToRequest(answers), [answers]);
  const queryOptions = {
    queryKey: ["model-recommendations", accessToken, body],
    queryFn: ({ signal }: { signal: AbortSignal }) => requestRecommendations(body, accessToken ?? "", signal),
    enabled: answers.step === "5" && Boolean(accessToken),
    retry: false,
    staleTime: 30000,
  };
  const query = useQuery(queryOptions);
  const result = !query.isFetching && !query.error ? query.data : undefined;
  const providerBody = useMemo(
    () => ({
      ...answersToRequest(EMPTY_ANSWERS),
      include_external: answers.include_external,
      priorities: { intelligence: 0, speed: 0, cost: 0 },
      limit: 1,
    }),
    [answers.include_external],
  );
  const providerOptions = {
    queryKey: ["model-recommendation-providers", accessToken, providerBody],
    queryFn: ({ signal }: { signal: AbortSignal }) => requestRecommendations(providerBody, accessToken ?? "", signal),
    enabled: answers.step === "4" && Boolean(accessToken),
    retry: false,
    staleTime: 30000,
  };
  const providerQuery = useQuery(providerOptions);
  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>{t("title")}</PageHeaderTitle>
        <PageHeaderDescription>{t("guidedIntro")}</PageHeaderDescription>
      </PageHeader>
      <Progress step={answers.step} />
      {answers.step === "5" ? (
        <>
          <Criteria answers={answers} onChange={setAnswers} />
          {query.isFetching && (
            <Panel title={t("finding")}>
              <Skeleton className="h-64 w-full rounded-xl" />
            </Panel>
          )}
          {query.error && <RequestError error={query.error} retry={() => void query.refetch()} />}
          {result && <Results result={result} />}
        </>
      ) : (
        <Questions
          answers={answers}
          onChange={setAnswers}
          providers={providerQuery.data?.available_providers ?? []}
          loading={providerQuery.isFetching}
          error={providerQuery.error}
          retry={() => void providerQuery.refetch()}
        />
      )}
    </Page>
  );
}

function RequestError({ error, retry }: { error: Error; retry: () => void }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  return (
    <Panel title={t("requestFailed")}>
      <p role="alert" className="mb-4 text-sm text-destructive">
        {error.message}
      </p>
      <Button variant="outline" onClick={retry}>
        {t("retry")}
      </Button>
    </Panel>
  );
}

function Questions({
  answers,
  onChange,
  providers,
  loading,
  error,
  retry,
}: {
  answers: Answers;
  onChange: ChangeAnswers;
  providers: readonly string[];
  loading: boolean;
  error: Error | null;
  retry: () => void;
}) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const title = { "1": t("question1"), "2": t("question2"), "3": t("question3"), "4": t("question4"), "5": "" }[
    answers.step
  ];
  const index = STEPS.indexOf(answers.step);
  return (
    <div className="flex flex-col gap-5 py-3">
      <Panel title={title} bodyClassName="gap-6">
        {answers.step === "1" && <PrioritySelectors answers={answers} onChange={onChange} />}
        {answers.step === "2" && <RequirementsSelectors answers={answers} onChange={onChange} />}
        {answers.step === "3" && <FocusSelectors answers={answers} onChange={onChange} />}
        {answers.step === "4" && (
          <>
            {loading && <Skeleton className="h-32 w-full" />}
            {error && <RequestError error={error} retry={retry} />}
            {!loading && !error && <ProviderSelectors answers={answers} onChange={onChange} providers={providers} />}
          </>
        )}
        <div className="mt-6 flex items-center justify-between gap-3 border-t pt-5">
          {answers.step === "1" ? (
            <Button variant="ghost" onClick={() => onChange(EMPTY_ANSWERS)}>
              {t("reset")}
            </Button>
          ) : (
            <Button variant="outline" onClick={() => onChange({ step: STEPS[index - 1] })}>
              <ArrowLeft />
              {t("back")}
            </Button>
          )}
          <Button
            disabled={answers.step === "3" && answers.focus.length === 0}
            onClick={() => onChange({ step: STEPS[index + 1] })}
          >
            {answers.step === "4" ? t("recommend") : t("continue")}
            <ArrowRight />
          </Button>
        </div>
      </Panel>
      <p className="text-center text-xs text-muted-foreground">{t("noInference")}</p>
    </div>
  );
}

function Criteria({ answers, onChange }: { answers: Answers; onChange: ChangeAnswers }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [shareStatus, setShareStatus] = useState<"idle" | "copied" | "failed">("idle");
  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShareStatus("copied");
    } catch {
      setShareStatus("failed");
    }
  };
  return (
    <Panel
      title={t("criteria")}
      action={
        <>
          <Button variant="outline" size="sm" onClick={() => onChange({ step: "1" })}>
            {t("editAnswers")}
          </Button>
          <Button variant="outline" size="sm" onClick={share}>
            <Share2 />
            {shareStatus === "copied" ? t("linkCopied") : t("share")}
          </Button>
        </>
      }
      bodyClassName="gap-4"
    >
      <PrioritySelectors answers={answers} />
      <label className="flex cursor-pointer items-center justify-between gap-4 text-sm">
        <span>{t("includeExternal")}</span>
        <Switch
          checked={answers.include_external}
          onCheckedChange={(include_external) => onChange({ include_external })}
        />
      </label>
      <div className="flex flex-wrap gap-2">
        {answers.focus.map((focus) => (
          <Badge key={focus} variant="secondary">
            {t(`building.${focus}`)}
          </Badge>
        ))}
        {answers.providers.map((provider) => (
          <Badge key={provider} variant="outline">
            {provider}
          </Badge>
        ))}
        {answers.include_external && <Badge variant="outline">{t("includeExternal")}</Badge>}
      </div>
      {shareStatus === "failed" && (
        <p role="alert" className="text-xs text-destructive">
          {t("shareFailed")}
        </p>
      )}
    </Panel>
  );
}

function Results({ result }: { result: RecommendationResponse }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [first, ...rest] = result.recommendations.slice(0, 3);
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
          <MoreOptions models={result.recommendations.slice(3, 10)} />
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
