"use client";

import { englishTranslate, type Translate } from "@/i18n/language";
import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdentityCell } from "@/components/shared/table_cells";
import { Tag } from "@/components/tag_management/types";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

export const DYNAMIC_SPEND_TAG_DESCRIPTION =
  "This is just a spend tag that was passed dynamically in a request. It does not control any LLM models.";

const isDynamicSpendTag = (tag: Tag) => tag.description === DYNAMIC_SPEND_TAG_DESCRIPTION;

function TagNameCell({ tag, onSelectTag }: { tag: Tag; onSelectTag: (tagName: string) => void }) {
  const { t } = useLanguage();
  if (isDynamicSpendTag(tag)) {
    return (
      <CellTooltip
        content={t("You cannot view the information of a dynamically generated spend tag")}
        trigger={<span className="block max-w-60 truncate font-mono text-xs text-muted-foreground">{tag.name}</span>}
      />
    );
  }
  return (
    <IdentityCell
      title={tag.name}
      titleClassName="font-mono text-xs font-normal text-primary"
      className="max-w-60"
      onClick={() => onSelectTag(tag.name)}
    />
  );
}

function TagModelsCell({ tag }: { tag: Tag }) {
  const { t } = useLanguage();
  const models = tag.models ?? [];
  if (models.length === 0) {
    return <Badge variant="secondary">{t("All Models")}</Badge>;
  }
  return (
    <div className="flex flex-wrap items-center gap-1">
      {models.map((modelId) => (
        <CellTooltip
          key={modelId}
          content={`ID: ${modelId}`}
          trigger={
            <Badge variant="outline" className="cursor-default">
              {tag.model_info?.[modelId] || modelId}
            </Badge>
          }
        />
      ))}
    </div>
  );
}

interface TagRowActionsProps {
  tag: Tag;
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
}

function TagRowActions({ tag, onEdit, onDelete }: TagRowActionsProps) {
  const { t } = useLanguage();
  const isDynamic = isDynamicSpendTag(tag);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open tag actions")}
        data-testid={`tag-actions-${tag.name}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          disabled={isDynamic}
          data-testid="tag-action-edit"
          title={isDynamic ? t("Dynamically generated spend tags cannot be edited") : undefined}
          onClick={() => onEdit(tag)}
        >
          <Pencil />
          {t("Edit")}
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          disabled={isDynamic}
          data-testid="tag-action-delete"
          title={isDynamic ? t("Dynamically generated spend tags cannot be deleted") : undefined}
          onClick={() => onDelete(tag.name)}
        >
          <Trash2 />
          {t("Delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface TagTableColumnsDeps {
  t?: Translate;
  onSelectTag: (tagName: string) => void;
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
}

export const getTagTableColumns = ({
  t = englishTranslate,
  onSelectTag,
  onEdit,
  onDelete,
}: TagTableColumnsDeps): ColumnDef<Tag>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: t("Tag Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Tag Name")} />,
    size: 260,
    enableSorting: true,
    filterFn: "includesString",
    cell: ({ row }) => <TagNameCell tag={row.original} onSelectTag={onSelectTag} />,
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: t("Description") },
    header: t("Description"),
    size: 300,
    enableSorting: false,
    filterFn: "includesString",
    cell: ({ row }) => {
      const description = isDynamicSpendTag(row.original) ? t(DYNAMIC_SPEND_TAG_DESCRIPTION) : row.original.description;
      return (
        <span className="block max-w-72 truncate text-sm text-muted-foreground" title={description}>
          {description || "-"}
        </span>
      );
    },
  },
  {
    id: "models",
    meta: { title: t("Allowed Models"), skeleton: "chips" },
    header: t("Allowed Models"),
    size: 240,
    enableSorting: false,
    cell: ({ row }) => <TagModelsCell tag={row.original} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: t("Created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("Actions")}</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <TagRowActions tag={row.original} onEdit={onEdit} onDelete={onDelete} />
      </div>
    ),
  },
];
