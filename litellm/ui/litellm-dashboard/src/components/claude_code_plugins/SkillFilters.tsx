import { Search } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ScanFilter } from "./skillCatalog";

interface SkillFiltersProps {
  search: string;
  tag: string;
  scan: ScanFilter;
  tags: string[];
  active: boolean;
  setSearch: (value: string) => void;
  setTag: (value: string) => void;
  setScan: (value: ScanFilter) => void;
  clear: () => void;
}

export function SkillFilters({
  search,
  tag,
  scan,
  tags,
  active,
  setSearch,
  setTag,
  setScan,
  clear,
}: SkillFiltersProps) {
  const { t } = useTranslation();
  const tagItems = [
    { value: "", label: t("skills.filters.allTags") },
    ...[...new Set([...tags, ...(tag ? [tag] : [])])].map((value) => ({ value, label: value })),
  ];
  const scanItems = [
    { value: "all", label: t("skills.filters.allScans") },
    { value: "yes", label: t("skills.filters.scanned") },
    { value: "no", label: t("skills.filters.notScanned") },
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      <InputGroup className="w-full sm:w-80">
        <InputGroupAddon>
          <Search className="size-4" />
        </InputGroupAddon>
        <InputGroupInput
          aria-label={t("skills.filters.search")}
          placeholder={t("skills.filters.search")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </InputGroup>
      <Select items={tagItems} value={tag} onValueChange={(value) => setTag(value ?? "")}>
        <SelectTrigger className="w-44" aria-label={t("skills.filters.tag")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {tagItems.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        items={scanItems}
        value={scan}
        onValueChange={(value) => setScan(value === "yes" || value === "no" ? value : "all")}
      >
        <SelectTrigger className="w-48" aria-label={t("skills.table.columns.securityScan")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {scanItems.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {active && (
        <Button variant="ghost" size="sm" onClick={clear}>
          {t("skills.filters.clear")}
        </Button>
      )}
    </div>
  );
}
