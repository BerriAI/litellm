"use client";

import { useMemo } from "react";
import { ArrowRight, Box, Plug, Search, Sparkles, Zap } from "lucide-react";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { uiHref } from "@/utils/uiHref";
import {
  availableProviders,
  browseModels,
  formatPerMillion,
  formatPublishedOn,
  formatTokens,
  KIND_LABEL,
  LAUNCH_KINDS,
  LAUNCHES,
  toCatalog,
  visibleLaunches,
  type LaunchKind,
} from "./discoverContent";

const KIND_ICON: Record<LaunchKind, typeof Box> = { model: Box, provider: Plug, feature: Zap };
const KIND_FILTERS = ["all", ...LAUNCH_KINDS] as const;
const KIND_FILTER_LABEL: Record<(typeof KIND_FILTERS)[number], string> = {
  all: "All",
  model: "Models",
  provider: "Providers",
  feature: "Features",
};

export const discoverQueryParsers = {
  kind: parseAsStringLiteral(KIND_FILTERS).withDefault("all"),
  provider: parseAsString,
  q: parseAsString.withDefault(""),
};

export default function DiscoverPage() {
  const { accessToken } = useAuthorized();
  const { data: costMap } = useModelCostMap(Boolean(accessToken), true);
  const [{ kind, provider, q }, setQuery] = useQueryStates(discoverQueryParsers);

  const launches = useMemo(() => visibleLaunches(LAUNCHES, kind), [kind]);
  const catalog = useMemo(() => toCatalog(costMap), [costMap]);
  const providers = useMemo(() => availableProviders(catalog), [catalog]);
  const models = useMemo(() => browseModels(catalog, { query: q, provider }), [catalog, provider, q]);

  return (
    <div className="mx-4">
      <div className="mt-2 flex w-full flex-col gap-8 p-8">
        <div>
          <h2 className="text-lg font-semibold">Discover</h2>
          <p className="text-sm text-muted-foreground">What&apos;s new in LiteLLM, and models you can route to today.</p>
        </div>

        <section aria-labelledby="discover-whats-new" className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-muted-foreground" />
              <h3 id="discover-whats-new" className="text-sm font-semibold">
                What&apos;s new
              </h3>
            </div>
            <div className="flex gap-1" role="group" aria-label="Filter launches">
              {KIND_FILTERS.map((filter) => (
                <Button
                  key={filter}
                  size="sm"
                  variant={kind === filter ? "secondary" : "ghost"}
                  aria-pressed={kind === filter}
                  onClick={() => void setQuery({ kind: filter === "all" ? null : filter })}
                >
                  {KIND_FILTER_LABEL[filter]}
                </Button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            {launches.map((item) => {
              const Icon = KIND_ICON[item.kind];
              return (
                <a key={item.title} href={item.href} target="_blank" rel="noopener noreferrer">
                  <Card className="h-full cursor-pointer gap-2 px-4 py-3 hover:ring-foreground/25">
                    <div className="flex items-center justify-between">
                      <div className="flex size-8 items-center justify-center rounded-md bg-muted">
                        <Icon className="size-4" />
                      </div>
                      <Badge variant="outline">{KIND_LABEL[item.kind]}</Badge>
                    </div>
                    <div className="text-sm font-medium leading-tight">{item.title}</div>
                    <p className="m-0 line-clamp-2 text-xs text-muted-foreground">{item.description}</p>
                    <div className="mt-auto flex items-center justify-between text-xs text-muted-foreground">
                      <span>{formatPublishedOn(item.publishedOn)}</span>
                      <ArrowRight className="size-3.5" />
                    </div>
                  </Card>
                </a>
              );
            })}
          </div>
        </section>

        <section aria-labelledby="discover-models" className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Box className="size-4 text-muted-foreground" />
              <h3 id="discover-models" className="text-sm font-semibold">
                Discover models
              </h3>
              <span className="text-xs text-muted-foreground">
                {catalog.length.toLocaleString()} in the LiteLLM catalog
              </span>
            </div>
            <div className="relative w-72">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-8"
                aria-label="Search models"
                placeholder="Search models"
                value={q}
                onChange={(e) => void setQuery({ q: e.target.value || null })}
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by provider">
            <Button
              size="sm"
              variant={provider === null ? "secondary" : "ghost"}
              aria-pressed={provider === null}
              onClick={() => void setQuery({ provider: null })}
            >
              All providers
            </Button>
            {providers.map((p) => (
              <Button
                key={p}
                size="sm"
                variant={provider === p ? "secondary" : "ghost"}
                aria-pressed={provider === p}
                onClick={() => void setQuery({ provider: p })}
              >
                {getProviderLogoAndName(p).displayName}
              </Button>
            ))}
          </div>
          <ul className="m-0 grid list-none grid-cols-1 gap-3 p-0 md:grid-cols-2 xl:grid-cols-4">
            {models.map((m) => {
              const { logo, displayName } = getProviderLogoAndName(m.provider);
              return (
                <li key={m.name}>
                  <Card className="gap-2 px-4 py-3">
                    <div className="flex items-center gap-2">
                      {logo ? <img src={logo} alt="" className="size-5 rounded" /> : <Box className="size-5" />}
                      <span className="truncate text-xs text-muted-foreground">{displayName}</span>
                      <Badge variant="outline" className="ml-auto">
                        {m.mode}
                      </Badge>
                    </div>
                    <div className="truncate text-sm font-medium" title={m.name}>
                      {m.name}
                    </div>
                    <dl className="m-0 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                      <div>
                        <dt className="text-[10px] uppercase">Context</dt>
                        <dd className="m-0 text-foreground">{formatTokens(m.contextWindow)}</dd>
                      </div>
                      <div>
                        <dt className="text-[10px] uppercase">In / 1M</dt>
                        <dd className="m-0 text-foreground">{formatPerMillion(m.inputCostPerToken)}</dd>
                      </div>
                      <div>
                        <dt className="text-[10px] uppercase">Out / 1M</dt>
                        <dd className="m-0 text-foreground">{formatPerMillion(m.outputCostPerToken)}</dd>
                      </div>
                    </dl>
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-1 w-full"
                      nativeButton={false}
                      role="link"
                      render={<a href={`${uiHref("models-and-endpoints")}?tab=add`} />}
                    >
                      Add to proxy
                      <ArrowRight />
                    </Button>
                  </Card>
                </li>
              );
            })}
          </ul>
          <div>
            <Button
              variant="link"
              className="px-0"
              nativeButton={false}
              role="link"
              render={<a href={uiHref("model-hub-table")} />}
            >
              Browse the full AI Hub
              <ArrowRight />
            </Button>
          </div>
        </section>
      </div>
    </div>
  );
}
