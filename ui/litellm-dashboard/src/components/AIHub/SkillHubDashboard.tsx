import React, { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search } from "lucide-react";
import { Plugin } from "@/components/claude_code_plugins/types";
import { ModelDataTable } from "@/components/model_dashboard/table";
import { skillHubColumns } from "@/components/skill_hub_table_columns";
import SkillDetail from "@/components/claude_code_plugins/skill_detail";

interface SkillHubDashboardProps {
  skills: Plugin[];
  isLoading: boolean;
  isAdmin?: boolean;
  accessToken?: string | null;
  publicPage?: boolean;
  onPublishSuccess?: () => void;
}

const ALL_DOMAINS = "__all__";

const SkillHubDashboard: React.FC<SkillHubDashboardProps> = ({
  skills,
  isLoading,
  isAdmin,
  accessToken,
  publicPage = false,
  onPublishSuccess,
}) => {
  const [search, setSearch] = useState("");
  const [domainFilter, setDomainFilter] = useState<string | undefined>(
    undefined,
  );
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const totalSkills = skills.length;
  const domains = useMemo(
    () => [...new Set(skills.map((s) => s.domain).filter(Boolean))],
    [skills],
  );
  const namespaces = useMemo(
    () => [...new Set(skills.map((s) => s.namespace).filter(Boolean))],
    [skills],
  );

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

  if (isLoading) {
    return (
      <div className="text-center py-16 text-muted-foreground">
        Loading skills...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Total Skills</div>
          <div className="text-2xl font-semibold text-foreground">
            {totalSkills}
          </div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Namespaces</div>
          <div className="text-2xl font-semibold text-foreground">
            {namespaces.length}
          </div>
        </div>
        <div className="border border-border rounded-lg p-4">
          <div className="text-xs text-muted-foreground mb-1">Domains</div>
          <div className="text-2xl font-semibold text-foreground">
            {domains.length}
          </div>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground">
            All {publicPage ? "Public " : ""}Skills
          </h3>
          <div className="flex items-center gap-2">
            <Select
              value={domainFilter ?? ALL_DOMAINS}
              onValueChange={(v) =>
                setDomainFilter(v === ALL_DOMAINS ? undefined : v)
              }
            >
              <SelectTrigger className="w-40">
                <SelectValue placeholder="All Domains" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_DOMAINS}>All Domains</SelectItem>
                {domains.map((d) => (
                  <SelectItem key={d!} value={d!}>
                    {d}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="relative w-72">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
              <Input
                placeholder="Search by name, namespace, or tag…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-8 h-9"
              />
            </div>
          </div>
        </div>
        <ModelDataTable
          columns={skillHubColumns(
            (skill) => setSelectedSkill(skill),
            copyToClipboard,
            publicPage,
          )}
          data={filteredSkills}
          isLoading={false}
          defaultSorting={[{ id: "name", desc: false }]}
        />
        <div className="mt-3 text-center">
          <span className="text-sm text-muted-foreground">
            Showing {filteredSkills.length} of {totalSkills} skill
            {totalSkills !== 1 ? "s" : ""}
          </span>
        </div>
      </div>
    </div>
  );
};

export default SkillHubDashboard;
