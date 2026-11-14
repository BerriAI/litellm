"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";
import { PolicyAttachment } from "@/components/policies/types";

import { getAttachmentTableColumns } from "./AttachmentTableColumns";

interface AttachmentTableProps {
  attachments: PolicyAttachment[];
  isLoading: boolean;
  onDeleteClick: (attachmentId: string) => void;
  isAdmin: boolean;
  accessToken: string | null;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("policies.attachmentTable.empty.title")}</div>
      <div className="text-sm text-muted-foreground">{t("policies.attachmentTable.empty.description")}</div>
    </div>
  );
}

const AttachmentTable: React.FC<AttachmentTableProps> = ({
  attachments,
  isLoading,
  onDeleteClick,
  isAdmin,
  accessToken,
}) => {
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => {
    const deps = { isAdmin, accessToken, onDeleteClick, t };
    return getAttachmentTableColumns(deps);
  }, [isAdmin, accessToken, onDeleteClick, t]);

  return (
    <DataTable
      data={attachments}
      paginationMode="client"
      columns={columns}
      getRowId={(row) => row.attachment_id}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("policies.attachmentTable.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default AttachmentTable;
