import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import React, { useEffect, useState } from "react";
import { getPromptVersions, PromptSpec } from "../../networking";

interface VersionHistorySidePanelProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
  promptId: string;
  activeVersionId?: string;
  onSelectVersion?: (version: PromptSpec) => void;
}

const VersionHistorySidePanel: React.FC<VersionHistorySidePanelProps> = ({
  isOpen,
  onClose,
  accessToken,
  promptId,
  activeVersionId,
  onSelectVersion,
}) => {
  const [versions, setVersions] = useState<PromptSpec[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchVersions = async () => {
      setLoading(true);
      try {
        const basePromptId = promptId.includes(".v")
          ? promptId.split(".v")[0]
          : promptId;
        const response = await getPromptVersions(accessToken!, basePromptId);
        setVersions(response.prompts);
      } catch (error) {
        console.error("Error fetching prompt versions:", error);
      } finally {
        setLoading(false);
      }
    };

    if (isOpen && accessToken && promptId) {
      fetchVersions();
    }
  }, [isOpen, accessToken, promptId]);

  const getVersionNumber = (prompt: PromptSpec) => {
    if (prompt.version) {
      return `v${prompt.version}`;
    }

    const versionedId =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (prompt.litellm_params as any)?.prompt_id || prompt.prompt_id;
    if (versionedId.includes(".v")) {
      return `v${versionedId.split(".v")[1]}`;
    }
    if (versionedId.includes("_v")) {
      return `v${versionedId.split("_v")[1]}`;
    }
    return "v1";
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    return new Date(dateString).toLocaleString();
  };

  return (
    <Sheet
      open={isOpen}
      onOpenChange={(o) => (!o ? onClose() : undefined)}
      modal={false}
    >
      <SheetContent
        side="right"
        className="w-[400px] sm:max-w-[400px]"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <SheetHeader>
          <SheetTitle>Version History</SheetTitle>
        </SheetHeader>
        {loading ? (
          <div className="space-y-3 mt-4">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : versions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No version history available.
          </div>
        ) : (
          <div className="mt-4 overflow-y-auto">
            {versions.map((item, index) => {
              const itemVersionNum =
                item.version ||
                parseInt(getVersionNumber(item).replace("v", ""));

              let activeVersionNum: number | null = null;
              if (activeVersionId) {
                if (activeVersionId.includes(".v")) {
                  activeVersionNum = parseInt(activeVersionId.split(".v")[1]);
                } else if (activeVersionId.includes("_v")) {
                  activeVersionNum = parseInt(activeVersionId.split("_v")[1]);
                }
              }

              const isSelected = activeVersionNum
                ? itemVersionNum === activeVersionNum
                : index === 0;

              return (
                <div
                  key={`${item.prompt_id}-v${item.version || itemVersionNum}`}
                  className={cn(
                    "mb-4 p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md",
                    isSelected
                      ? "border-primary bg-primary/10"
                      : "border-border bg-background hover:border-primary/50",
                  )}
                  onClick={() => onSelectVersion?.(item)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="m-0">
                        {getVersionNumber(item)}
                      </Badge>
                      {index === 0 && (
                        <Badge className="m-0 bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                          Latest
                        </Badge>
                      )}
                    </div>
                    {isSelected && (
                      <Badge className="m-0 bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                        Active
                      </Badge>
                    )}
                  </div>

                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-muted-foreground font-medium">
                      {formatDate(item.created_at)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {item.prompt_info?.prompt_type === "db"
                        ? "Saved to Database"
                        : "Config Prompt"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
};

export default VersionHistorySidePanel;
