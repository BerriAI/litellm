"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { modelHubCall, PromptSpec } from "@/components/networking";

import { getPromptTableColumns } from "./PromptTableColumns";
import { ModelGroupInfo } from "./prompt_utils";

interface PromptTableProps {
  promptsList: PromptSpec[];
  isLoading: boolean;
  onPromptClick?: (id: string) => void;
  onDeleteClick?: (id: string, name: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No prompts yet</div>
      <div className="text-sm text-muted-foreground">Add a prompt to start managing reusable templates.</div>
    </div>
  );
}

const PromptTable: React.FC<PromptTableProps> = ({
  promptsList,
  isLoading,
  onPromptClick,
  onDeleteClick,
  accessToken,
  isAdmin,
}) => {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [modelHubData, setModelHubData] = useState<Map<string, ModelGroupInfo>>(new Map());

  useEffect(() => {
    const fetchModelHubData = async () => {
      if (!accessToken) return;

      try {
        const response = await modelHubCall(accessToken);
        if (response?.data) {
          const modelMap = new Map<string, ModelGroupInfo>();
          response.data.forEach((model: ModelGroupInfo) => {
            modelMap.set(model.model_group, model);
          });
          setModelHubData(modelMap);
        }
      } catch (error) {
        console.error("Error fetching model hub data:", error);
      }
    };

    fetchModelHubData();
  }, [accessToken]);

  const columns = useMemo(
    () => getPromptTableColumns({ modelHubData, isAdmin, onPromptClick, onDeleteClick }),
    [modelHubData, isAdmin, onPromptClick, onDeleteClick],
  );

  return (
    <DataTable
      data={promptsList}
      columns={columns}
      getRowId={(prompt, index) => prompt.prompt_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading prompts…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default PromptTable;
