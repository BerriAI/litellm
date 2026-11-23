"use client";

import { ArrowLeft, Clock, Ellipsis, KeyRound, Lock, RefreshCw, Search, User, Users, Wallet } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Tabs } from "@/components/ui/tabs";

import { middleTruncate } from "./formatters";
import { KeyTabsList } from "./KeyTabsList";
import { BadgeRow, Caption, FactLabel, SectionCard, SubHead, ValueSkeleton } from "./SectionCard";

interface KeyDetailSkeletonProps {
  keyId: string;
  backButtonText: string;
  onBack: () => void;
  showAutoRouter: boolean;
}

const MODEL_WIDTHS = [110, 140, 90, 160, 120];
const TIMELINE_WIDTHS = [84, 78, 72, 66];

export function KeyDetailSkeleton({ keyId, backButtonText, onBack, showAutoRouter }: KeyDetailSkeletonProps) {
  const { t } = useTranslation();
  return (
    <div className="h-full w-full overflow-y-auto p-4 sm:p-6" aria-busy="true" data-testid="key-detail-skeleton">
      <div className="mx-auto flex max-w-[1180px] flex-col gap-6">
        <Tabs value="overview">
          <header>
            <button
              type="button"
              onClick={onBack}
              className="mb-2 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="size-3.5" />
              {backButtonText}
            </button>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex min-w-0 flex-col gap-1.5">
                <div className="flex items-center gap-2.5">
                  <h1 className="m-0 flex items-center gap-2 text-xl leading-7 font-semibold">
                    <KeyRound className="size-5 shrink-0" strokeWidth={1.75} aria-hidden />
                    <ValueSkeleton width={180} height={20} />
                  </h1>
                  <ValueSkeleton width={56} height={20} />
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] leading-5 text-muted-foreground">
                  <span className="inline-flex items-center gap-1.5">
                    <User className="size-4" aria-hidden />
                    <ValueSkeleton width={120} />
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <Users className="size-4" aria-hidden />
                    <ValueSkeleton width={72} />
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <ValueSkeleton width={64} height={18} />
                    <span className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground" title={keyId}>
                      {middleTruncate(keyId)}
                    </span>
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" disabled>
                  <RefreshCw className="size-3.5" />
                  {t("keyDetail.keyInfoHeader.regenerateKey")}
                </Button>
                <Button variant="outline" size="icon-sm" disabled aria-label={t("keyDetail.keyInfoHeader.moreActions")}>
                  <Ellipsis className="size-4" />
                </Button>
              </div>
            </div>
            <div className="mt-4">
              <KeyTabsList showAutoRouter={showAutoRouter} />
            </div>
          </header>
        </Tabs>

        <div className="flex flex-col gap-4">
          <SectionCard
            icon={Wallet}
            title={t("keyDetail.overview.spend.title")}
            action={
              <Button variant="outline" size="sm" disabled>
                {t("keyDetail.overview.spend.openActivity")}
              </Button>
            }
          >
            <div className="grid grid-cols-1 items-start gap-6 md:grid-cols-[minmax(180px,240px)_minmax(0,1fr)]">
              <div className="flex flex-col gap-3">
                <div>
                  <div className="text-xs font-medium text-muted-foreground">
                    <ValueSkeleton width={64} height={10} />
                  </div>
                  <p className="m-0 mt-1 text-2xl leading-8 font-semibold">
                    <ValueSkeleton width={120} height={24} />
                  </p>
                  <p className="m-0 text-xs">
                    <ValueSkeleton width={150} height={10} />
                  </p>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground">
                    {t("keyDetail.overview.spend.lifetime")}
                  </div>
                  <p className="m-0 mt-1 text-[15px] leading-[22px] font-semibold">
                    <ValueSkeleton width={90} height={16} />
                  </p>
                  <p className="m-0 text-xs">
                    <ValueSkeleton width={90} height={10} />
                  </p>
                </div>
              </div>
              <div className="min-w-0">
                <div className="mb-2 text-xs font-medium text-muted-foreground">
                  {t("keyDetail.overview.canStillSpend.title")}
                </div>
                <div className="text-2xl leading-8 font-semibold">
                  <ValueSkeleton width={120} height={24} />
                </div>
                <p className="m-0 mt-1 mb-2">
                  <ValueSkeleton width={300} />
                </p>
                <div className="h-1.5 w-full animate-pulse rounded-full bg-muted" />
                <Caption className="mt-1">
                  <ValueSkeleton width={110} height={10} />
                </Caption>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <SubHead title={t("shared.appliedBudgets.title")} />
              <div className="grid grid-cols-1 items-center gap-x-6 gap-y-1 border-t border-border py-2.5 md:grid-cols-[minmax(160px,260px)_minmax(120px,1fr)_minmax(180px,220px)]">
                <div>
                  <div>
                    <ValueSkeleton width={90} />
                  </div>
                  <Caption>
                    <ValueSkeleton width={120} height={10} />
                  </Caption>
                </div>
                <div className="h-1.5 w-full animate-pulse rounded-full bg-muted" />
                <div className="md:text-right">
                  <div>
                    <ValueSkeleton width={130} />
                  </div>
                  <Caption>
                    <ValueSkeleton width={70} height={10} />
                  </Caption>
                </div>
              </div>
            </div>
          </SectionCard>

          <SectionCard icon={Clock} title={t("keyDetail.overview.lifecycle.title")}>
            <ol className="m-0 grid list-none grid-cols-2 gap-y-4 p-0 sm:grid-flow-col sm:auto-cols-fr sm:grid-cols-none sm:gap-y-0">
              {TIMELINE_WIDTHS.map((width, index) => (
                <li key={width} className="relative min-w-0 pt-[18px] pr-4">
                  <span aria-hidden className="absolute top-0 left-0 size-[9px] animate-pulse rounded-full bg-muted" />
                  <FactLabel>
                    <ValueSkeleton width={56} height={10} />
                  </FactLabel>
                  <div className="mt-0.5 font-medium">
                    <ValueSkeleton width={width} />
                  </div>
                  {index === 0 ? (
                    <Caption>
                      <ValueSkeleton width={80} height={10} />
                    </Caption>
                  ) : null}
                </li>
              ))}
            </ol>
          </SectionCard>

          <SectionCard icon={Lock} title={t("keyDetail.overview.access.title")}>
            <div className="flex flex-col gap-2">
              <SubHead title={t("keyDetail.overview.access.models")} />
              <BadgeRow>
                {MODEL_WIDTHS.map((width) => (
                  <ValueSkeleton key={width} width={width} height={20} />
                ))}
              </BadgeRow>
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-x-6 gap-y-4 border-t border-border pt-3">
              <div className="flex min-w-0 flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <Search className="size-4 text-muted-foreground" aria-hidden />
                  <span className="font-medium">{t("keyDetail.overview.access.searchTools")}</span>
                </div>
                <div>
                  <ValueSkeleton width={80} />
                </div>
              </div>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
