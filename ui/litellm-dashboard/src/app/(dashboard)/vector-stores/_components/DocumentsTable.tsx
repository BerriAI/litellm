"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { DocumentUpload } from "@/components/vector_store_management/types";

import { getDocumentsTableColumns } from "./DocumentsTableColumns";

interface DocumentsTableProps {
  documents: DocumentUpload[];
  onRemove: (uid: string) => void;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No documents uploaded yet</div>
      <div className="text-sm text-muted-foreground">Upload documents above to get started.</div>
    </div>
  );
}

const DocumentsTable: React.FC<DocumentsTableProps> = ({ documents, onRemove }) => {
  const columns = useMemo(() => getDocumentsTableColumns({ onRemove }), [onRemove]);

  return (
    <DataTable
      data={documents}
      columns={columns}
      getRowId={(document, index) => document.uid || String(index)}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default DocumentsTable;
