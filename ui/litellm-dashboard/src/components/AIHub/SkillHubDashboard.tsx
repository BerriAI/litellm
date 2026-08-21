import React, { useMemo, useState } from "react";
import { SortingState } from "@tanstack/react-table";
import { Inbox, Search, X } from "lucide-react";
import { Plugin } from "@/components/claude_code_plugins/types";
import { DataTable } from "@/components/shared/DataTable";
import { getSkillHubTableColumns } from "@/components/AIHub/SkillHubTableColumns";
import SkillDetail from "@/components/claude_code_plugins/skill_detail";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ALL_DOMAINS = "__all_domains__";

interface SkillHubDashboardProps {
  skills: Plugin[];
  isLoading: boolean;
  isAdmin?: boolean;
  accessToken?: string | null;
  publicPage?: boolean;
  onPublishSuccess?: () => void;
}

function SkillsEmptyState({ filtered }: { filtered: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{filtered ? "No matching skills" : "No skills yet"}</div>
      <div className="text-sm text-muted-foreground">
        {filtered
          ? "Adjust the search or domain filter to see more skills."
          : "Skills added here will appear for developers."}
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
  const [search, setSearch] = useState("");
  const [domainFilter, setDomainFilter] = useState<string | undefined>(undefined);
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);
  const [sorting, setSorting] = useState<SortingState>([{ id: "name", desc: false }]);

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
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.description?.toLowerCase().includes(q) ||
          s.domain?.toLowerCase().includes(q) ||
          s.namespace?.toLowerCase().includes(q) ||
          s.keywords?.some((k) => k.toLowerCase().includes(q)),
      );
    }
    return result;
  }, [skills, search, domainFilter]);

  const columns = useMemo(() => getSkillHubTableColumns({ onSkillClick: setSelectedSkill }), []);

  const domainItems = useMemo(
    () => [{ value: ALL_DOMAINS, label: "All Domains" }, ...domains.map((d) => ({ value: d, label: d }))],
    [domains],
  );

  const hasActiveFilter = search.trim().length > 0 || domainFilter != null;

  if (selectedSkill) {
    return (
      <SkillDetail
        skill={selectedSkill}
        onBack={() => setSelectedSkill(null)}
        isAdmin={isAdmin}
        accessToken={accessToken}
        onPublishClick={onPublishSuccess}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Total Skills</div>
          <div className="text-2xl font-semibold text-foreground">{totalSkills}</div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Namespaces</div>
          <div className="text-2xl font-semibold text-foreground">{namespaces.length}</div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Domains</div>
          <div className="text-2xl font-semibold text-foreground">{domains.length}</div>
        </div>
      </div>

      {/* Search + filters + table */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">All {publicPage ? "Public " : ""}Skills</h3>
          <div className="flex items-center gap-2">
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
            <InputGroup className="w-[280px]">
              <InputGroupAddon>
                <Search className="size-4 text-muted-foreground" />
              </InputGroupAddon>
              <InputGroupInput
                placeholder="Search by name, namespace, or tag…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search !== "" && (
                <InputGroupAddon align="inline-end">
                  <InputGroupButton
                    size="icon-xs"
                    variant="ghost"
                    aria-label="Clear search"
                    onClick={() => setSearch("")}
                  >
                    <X className="size-3.5" />
                  </InputGroupButton>
                </InputGroupAddon>
              )}
            </InputGroup>
          </div>
        </div>
        <DataTable
          data={filteredSkills}
          columns={columns}
          getRowId={(skill, index) => skill.id || String(index)}
          sortingMode="client"
          sorting={sorting}
          onSortingChange={setSorting}
          isLoading={isLoading}
          loadingMessage="Loading skills…"
          noDataMessage={<SkillsEmptyState filtered={hasActiveFilter} />}
          size="compact"
        />
        <div className="mt-3 text-center">
          <p className="text-sm text-muted-foreground">
            Showing {filteredSkills.length} of {totalSkills} skill{totalSkills !== 1 ? "s" : ""}
          </p>
        </div>
      </div>
    </div>
  );
};

export default SkillHubDashboard;
