import { DataTableSortHeader } from "@/components/shared/DataTable/DataTableSortHeader";
import type { ColumnDef } from "@tanstack/react-table";
import { DateCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import i18n from "@/i18n/i18n";
import type { SkillCatalogMetadata } from "./skillCatalog";

const IDENTITY_LABELS = {
  created_by: "skills.table.columns.createdBy",
  approved_by: "skills.table.columns.approvedBy",
} as const;

export function getSkillMetadataColumns<T extends SkillCatalogMetadata>(): ColumnDef<T>[] {
  return [
    {
      id: "download_count",
      accessorKey: "download_count",
      header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t("skills.table.columns.downloads")} />,
      size: 120,
      sortUndefined: "last",
      accessorFn: (skill) => skill.download_count ?? undefined,
      cell: ({ row }) => (
        <span className="tabular-nums">{row.original.download_count?.toLocaleString(i18n.language) ?? "—"}</span>
      ),
    },
    {
      id: "keywords",
      header: i18n.t("skills.table.columns.tags"),
      size: 180,
      enableSorting: false,
      cell: ({ row }) => (
        <div className="flex max-w-52 flex-wrap gap-1">
          {row.original.keywords?.length
            ? row.original.keywords.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))
            : "—"}
        </div>
      ),
    },
    ...(["created_by", "approved_by"] as const).map(
      (field): ColumnDef<T> => ({
        id: field,
        accessorKey: field,
        size: 190,
        enableSorting: true,
        header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t(IDENTITY_LABELS[field])} />,
        cell: ({ row }) => (
          <span className="block max-w-48 truncate text-xs" title={row.original[field] ?? undefined}>
            {row.original[field] || i18n.t("skills.table.notRecorded")}
          </span>
        ),
      }),
    ),
    ...(["created_at", "updated_at"] as const).map(
      (field): ColumnDef<T> => ({
        id: field,
        accessorKey: field,
        size: 140,
        enableSorting: true,
        accessorFn: (skill) => (skill[field] ? new Date(skill[field]!).getTime() : undefined),
        sortUndefined: "last",
        header: ({ column }) => (
          <DataTableSortHeader
            column={column}
            title={i18n.t(field === "created_at" ? "skills.table.columns.createdAt" : "skills.table.columns.updatedAt")}
          />
        ),
        cell: ({ row }) => <DateCell value={row.original[field]} precision="date" />,
      }),
    ),
    {
      id: "security_scan_passed",
      accessorKey: "security_scan_passed",
      size: 130,
      enableSorting: true,
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={<span title={i18n.t("skills.table.scanHelp")}>{i18n.t("skills.table.columns.securityScan")}</span>}
        />
      ),
      cell: ({ row }) => (
        <span title={i18n.t("skills.table.scanHelp")}>
          <StatusBadge
            tone="neutral"
            label={i18n.t(row.original.security_scan_passed ? "skills.table.publicYes" : "skills.table.publicNo")}
          />
        </span>
      ),
    },
  ];
}
