"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { Puzzle } from "lucide-react";
import * as React from "react";
import { useTranslation } from "react-i18next";

import type { Plugin } from "@/components/claude_code_plugins/types";
import { DataTableSortHeader } from "@/components/shared/DataTable/DataTableSortHeader";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { SkillFilters } from "@/components/claude_code_plugins/SkillFilters";
import { getSkillMetadataColumns } from "@/components/claude_code_plugins/SkillMetadataColumns";
import { useSkillFilters, matchesSkillFilters, skillTags } from "@/components/claude_code_plugins/skillCatalog";

import { CatalogDetails } from "./CatalogDetails";
import { CatalogFilters } from "./CatalogFilters";
import { LoadFailed, NoResults, NotListed } from "./HubEmpty";
import { SkillInspector } from "./SkillInspector";
import { useOpenHref } from "./useHubUrlState";

interface SkillsPaneProps {
  rows: Plugin[] | undefined;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
  selected: string | null;
  onSelect: (key: string | null) => void;
}

function SkillLink({
  skill,
  selected,
  onSelect,
}: {
  skill: Plugin;
  selected: string | null;
  onSelect: (key: string | null) => void;
}) {
  const hrefFor = useOpenHref();
  const follow = (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.defaultPrevented) return;
    if (event.metaKey || event.ctrlKey) return;
    if (event.shiftKey || event.altKey) return;
    event.preventDefault();
    onSelect(skill.name);
  };
  return (
    <a
      href={hrefFor(skill.name)}
      aria-current={selected === skill.name ? "true" : undefined}
      className="flex min-w-0 flex-col gap-1 rounded-sm py-2 outline-none hover:text-primary focus-visible:ring-2 focus-visible:ring-ring"
      onClick={follow}
    >
      <span className="font-mono text-sm font-medium whitespace-normal wrap-anywhere">{skill.name}</span>
      {skill.description && (
        <span className="line-clamp-2 text-xs whitespace-normal text-muted-foreground">{skill.description}</span>
      )}
      {skill.domain && <span className="text-xs whitespace-normal text-muted-foreground">{skill.domain}</span>}
    </a>
  );
}

export function SkillsPane({ rows, loading, error, onRetry, selected, onSelect }: SkillsPaneProps) {
  const { t } = useTranslation();
  const filters = useSkillFilters();
  const query = filters.search;
  const [domains, setDomains] = React.useState<string[]>([]);
  const skills = React.useMemo(() => [...(rows ?? [])].sort((a, b) => a.name.localeCompare(b.name)), [rows]);
  const options = React.useMemo(
    () =>
      [...new Set(skills.flatMap((skill) => (skill.domain ? [skill.domain] : [])))].sort().map((domain) => ({
        domain,
        count: skills.filter((skill) => skill.domain === domain).length,
      })),
    [skills],
  );
  const shown = React.useMemo(
    () =>
      skills.filter((skill) => {
        const matchesQuery = matchesSkillFilters(skill, {
          search: filters.search,
          tag: filters.tag,
          scan: filters.scan,
        });
        const matchesDomain = domains.length === 0 || domains.includes(skill.domain ?? "");
        return matchesQuery && matchesDomain;
      }),
    [skills, filters.search, filters.tag, filters.scan, domains],
  );
  const opened = skills.find((skill) => skill.name === selected);
  const missingName = opened === undefined && !loading ? selected : null;
  const showRefreshError = error && !loading && shown.length > 0;
  const filtered = filters.active || domains.length > 0;
  const clear = () => {
    filters.clear();
    setDomains([]);
  };
  const columns: ColumnDef<Plugin>[] = [
    {
      accessorKey: "name",
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("publicModelHub.catalog.skill", { defaultValue: "Skill" })} />
      ),
      meta: { skeleton: "twoLine", className: "min-w-48 whitespace-normal" },
      cell: ({ row }) => <SkillLink skill={row.original} selected={selected} onSelect={onSelect} />,
    },
    ...getSkillMetadataColumns<Plugin>(),
  ];
  const countOptions = {
    count: shown.length,
    defaultValue_one: "{{count}} skill",
    defaultValue_other: "{{count}} skills",
    defaultValue: "{{count}} skills",
  };

  return (
    <div data-slot="catalog-layout" className="grid min-w-0 gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
      <CatalogFilters count={domains.length} onClear={() => setDomains([])}>
        <fieldset data-slot="catalog-filter-section" className="min-w-0">
          <legend className="mb-3 text-sm font-medium">
            {t("publicModelHub.catalog.domain", { defaultValue: "Domain" })}
          </legend>
          <div className="flex flex-col gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-between"
              aria-pressed={domains.length === 0}
              onClick={() => setDomains([])}
            >
              {t("publicModelHub.catalog.allDomains", { defaultValue: "All domains" })}
              <span className="text-xs text-muted-foreground tabular-nums">{loading ? "-" : skills.length}</span>
            </Button>
            {options.map(({ domain, count }) => (
              <label key={domain} className="flex cursor-pointer items-center gap-2.5 text-sm">
                <Checkbox
                  checked={domains.includes(domain)}
                  onCheckedChange={(checked) =>
                    setDomains((current) =>
                      checked ? [...current, domain] : current.filter((value) => value !== domain),
                    )
                  }
                />
                <span className="min-w-0 flex-1 wrap-anywhere">{domain}</span>
                <span aria-hidden="true" className="text-xs text-muted-foreground tabular-nums">
                  {count}
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      </CatalogFilters>
      <div data-slot="catalog-results" className="flex min-w-0 flex-col gap-4">
        <div data-slot="catalog-toolbar" className="flex flex-wrap items-center gap-3">
          <SkillFilters {...filters} tags={skillTags(skills)} active={filtered} clear={clear} />
        </div>
        <div data-slot="catalog-count" className="text-sm text-muted-foreground tabular-nums" aria-live="polite">
          {loading
            ? t("publicModelHub.catalog.loadingSkills", { defaultValue: "Loading skills…" })
            : !error && t("publicModelHub.catalog.skillCount", countOptions)}
        </div>
        {showRefreshError && <LoadFailed noun={t("publicModelHub.skillsPane.noun")} onRetry={onRetry} />}
        <div data-slot="catalog-table" aria-busy={loading}>
          <DataTable
            data={shown}
            sortingMode="client"
            sorting={filters.sorting}
            onSortingChange={filters.onSortingChange}
            columns={columns}
            getRowId={(skill) => skill.name}
            isLoading={loading}
            loadingMessage={t("publicModelHub.catalog.loadingSkills", { defaultValue: "Loading skills…" })}
            isError={error}
            errorMessage={t("publicModelHub.loadFailed.title", { noun: t("publicModelHub.skillsPane.noun") })}
            onRetry={onRetry}
            rowClassName={(row) => (row.original.name === selected ? "bg-muted/50" : "")}
            noDataMessage={
              <NoResults
                icon={<Puzzle />}
                noun={t("publicModelHub.skillsPane.noun")}
                query={query}
                onClear={filtered ? clear : undefined}
              />
            }
          />
        </div>
      </div>
      <CatalogDetails name={selected} loading={loading && opened === undefined} onClose={() => onSelect(null)}>
        {opened && <SkillInspector skill={opened} />}
        {missingName !== null && !error && <NotListed name={missingName} />}
        {missingName !== null && error && <LoadFailed noun={t("publicModelHub.skillsPane.noun")} onRetry={onRetry} />}
      </CatalogDetails>
    </div>
  );
}
