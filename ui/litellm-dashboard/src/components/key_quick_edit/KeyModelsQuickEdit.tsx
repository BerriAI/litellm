"use client";

import { useState } from "react";
import { Loader2, Pencil } from "lucide-react";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { Button } from "@/components/ui/button";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { hasAllModelsSentinel } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { modelSentinelOptions } from "@/components/templates/keyEditFieldNormalizers";
import { modelsQuickEditPayload } from "./quickEditPayload";
import { useKeyAssignableModels } from "./useKeyAssignableModels";
import { useQuickKeyUpdate } from "./useQuickKeyUpdate";

interface KeyModelsQuickEditProps {
  keyData: KeyResponse;
  team: Team | null | undefined;
  accessToken: string | null;
  userId: string | null;
  userRole: string | null;
  canModify: boolean;
  canEditModels: boolean;
  onKeyDataUpdate?: (updated: Partial<KeyResponse>) => void;
  buttonClassName?: string;
}

export function KeyModelsQuickEdit({
  keyData,
  team,
  accessToken,
  userId,
  userRole,
  canModify,
  canEditModels,
  onKeyDataUpdate,
  buttonClassName,
}: KeyModelsQuickEditProps) {
  const [open, setOpen] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>(keyData.models ?? []);
  const mutation = useQuickKeyUpdate(accessToken ?? "");
  const modelQuery = {
    keyData,
    team,
    userId,
    userRole,
    accessToken,
    enabled: open,
  };
  const { data: availableModels = [], isPending: modelsLoading } = useKeyAssignableModels(modelQuery);

  const hasRestrictedRoutes =
    keyData.allowed_routes?.some((route) => route === "management_routes" || route === "info_routes") ?? false;
  const hasEditPermission = canModify && canEditModels;
  const isAllowed = hasEditPermission && Boolean(accessToken) && !hasRestrictedRoutes;
  const modelOptions = [
    ...modelSentinelOptions(keyData.team_id, team != null),
    ...availableModels.map((model) => ({
      value: model,
      label: model,
      disabled: hasAllModelsSentinel(selectedModels),
    })),
  ];

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) setSelectedModels(keyData.models ?? []);
  };

  if (!isAllowed) return null;

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger
        render={
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            aria-label="Edit models"
            className={`opacity-0 transition-opacity group-hover/editable:opacity-100 focus-visible:opacity-100 ${buttonClassName ?? ""}`}
          >
            <Pencil aria-hidden="true" />
          </Button>
        }
      />
      <PopoverContent align="end" className="w-96 max-w-[calc(100vw-2rem)] gap-3">
        <PopoverTitle>Quick edit models</PopoverTitle>
        <MultiSelect
          options={modelOptions}
          value={selectedModels}
          onValueChange={setSelectedModels}
          placeholder="Select models"
          loading={modelsLoading}
        />
        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={mutation.isPending}
            onClick={() => {
              mutation.mutate(
                {
                  kind: "models",
                  payload: modelsQuickEditPayload(keyData.token || keyData.token_id, selectedModels),
                },
                {
                  onSuccess: (updated) => {
                    onKeyDataUpdate?.(updated);
                    setOpen(false);
                  },
                },
              );
            }}
          >
            {mutation.isPending && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
            Save
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
