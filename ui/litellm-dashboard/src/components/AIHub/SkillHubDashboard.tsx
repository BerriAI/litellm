import React, { useMemo, useState } from "react";
import { Inbox } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Plugin } from "@/components/claude_code_plugins/types";
import { DataTable } from "@/components/shared/DataTable";
import { getSkillHubTableColumns } from "@/components/AIHub/SkillHubTableColumns";
import SkillDetail, { legacySkillDetail } from "@/components/claude_code_plugins/skill_detail";
import { SkillFilters } from "@/components/claude_code_plugins/SkillFilters";
import {
  useSkillFilters,
  matchesSkillFilters,
  skillTags,
  SKILL_SORT_FIELDS,
} from "@/components/claude_code_plugins/skillCatalog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ALL_DOMAINS = "__all_domains__";
const DASHBOARD_SORT_FIELDS = [...SKILL_SORT_FIELDS, "category", "domain", "enabled"];

interface SkillHubDashboardProps {
  skills: Plugin[];
  isLoading: boolean;
  isAdmin?: boolean;
  accessToken?: string | null;
  publicPage?: boolean;
  onPublishSuccess?: () => void;
}

function SkillsEmptyState({ filtered }: { filtered: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {filtered
          ? t("aiHub.skillHubDashboard.emptyState.noMatchTitle")
          : t("aiHub.skillHubDashboard.emptyState.noneTitle")}
      </div>
      <div className="text-sm text-muted-foreground">
        {filtered
          ? t("aiHub.skillHubDashboard.emptyState.noMatchBody")
          : t("aiHub.skillHubDashboard.emptyState.noneBody")}
      </div>
    </div>
  );
}

const SkillHubDashboard: React.FC<SkillHubDashboardProps> = ({
  skills,
  isLoading,
  isAdmin,
  accessToken,
  publicPage = false,
  onPublishSuccess,
}) => {
  const { t } = useTranslation();
  const filters = useSkillFilters(DASHBOARD_SORT_FIELDS);
  const [domainFilter, setDomainFilter] = useState<string | undefined>(undefined);
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);

  // Derived stats
  const totalSkills = skills.length;
  const domains = useMemo(
    () => [...new Set(skills.map((s) => s.domain).filter((domain): domain is string => Boolean(domain)))],
    [skills],
  );
  const namespaces = useMemo(() => [...new Set(skills.map((s) => s.namespace).filter(Boolean))], [skills]);

  // Filtered table data
  const filteredSkills = useMemo(() => {
    let result = skills;
    if (domainFilter) {
      result = result.filter((s) => (s.domain || "General") === domainFilter);
    }
    return result.filter((skill) =>
      matchesSkillFilters(skill, { search: filters.search, tag: filters.tag, scan: filters.scan }),
    );
  }, [skills, filters.search, filters.tag, filters.scan, domainFilter]);

  const columns = useMemo(() => getSkillHubTableColumns({ onSkillClick: setSelectedSkill }), []);

  const domainItems = useMemo(
    () => [
      { value: ALL_DOMAINS, label: t("aiHub.skillHubDashboard.allDomains") },
      ...domains.map((d) => ({ value: d, label: d })),
    ],
    [domains, t],
  );

  const hasActiveFilter = filters.active || domainFilter != null;

  if (selectedSkill) {
    return (
      <SkillDetail
        skill={legacySkillDetail(selectedSkill)}
        onBack={() => setSelectedSkill(null)}
        isAdmin={isAdmin}
        accessToken={accessToken}
        onPublishClick={onPublishSuccess}
        onRestored={(restored) => {
          setSelectedSkill({ ...selectedSkill, ...restored, enabled: restored.visibility === "public" });
          onPublishSuccess?.();
        }}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">{t("aiHub.skillHubDashboard.totalSkills")}</div>
          <div className="text-2xl font-semibold text-foreground">{totalSkills}</div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">{t("aiHub.skillHubDashboard.namespaces")}</div>
          <div className="text-2xl font-semibold text-foreground">{namespaces.length}</div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">{t("aiHub.skillHubDashboard.domains")}</div>
          <div className="text-2xl font-semibold text-foreground">{domains.length}</div>
        </div>
      </div>

      {/* Search + filters + table */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">
            {publicPage ? t("aiHub.skillHubDashboard.allPublicSkills") : t("aiHub.skillHubDashboard.allSkills")}
          </h3>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              items={domainItems}
              value={domainFilter ?? ALL_DOMAINS}
              onValueChange={(val) => setDomainFilter(val === null || val === ALL_DOMAINS ? undefined : val)}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {domainItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <SkillFilters {...filters} tags={skillTags(skills)} />
          </div>
        </div>
        <DataTable
          data={filteredSkills}
          paginationMode="client"
          columns={columns}
          getRowId={(skill, index) => skill.id || String(index)}
          sortingMode="client"
          sorting={filters.sorting}
          onSortingChange={filters.onSortingChange}
          isLoading={isLoading}
          loadingMessage={t("aiHub.skillHubDashboard.loading")}
          noDataMessage={<SkillsEmptyState filtered={hasActiveFilter} />}
          size="compact"
        />
        <div className="mt-3 text-center">
          <p className="text-sm text-muted-foreground">
            {t("aiHub.skillHubDashboard.showing", { shown: filteredSkills.length, count: totalSkills })}
          </p>
        </div>
      </div>
    </div>
  );
};

export default SkillHubDashboard;
