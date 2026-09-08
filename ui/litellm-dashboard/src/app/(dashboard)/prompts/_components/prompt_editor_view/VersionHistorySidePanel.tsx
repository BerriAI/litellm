import React, { useEffect, useState } from "react";
import { getPromptVersions, PromptSpec } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { XIcon } from "lucide-react";

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
    if (isOpen && accessToken && promptId) {
      fetchVersions();
    }
  }, [isOpen, accessToken, promptId]);

  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (event: KeyboardEvent) => {
      const openModal = document.querySelector('[data-slot="dialog-content"][data-open]');
      if (event.key === "Escape" && !openModal) onClose();
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  const fetchVersions = async () => {
    setLoading(true);
    try {
      // Strip .v suffix if present to get base ID for querying all versions
      const basePromptId = promptId.includes(".v") ? promptId.split(".v")[0] : promptId;
      const response = await getPromptVersions(accessToken!, basePromptId);
      setVersions(response.prompts);
    } catch (error) {
      console.error("Error fetching prompt versions:", error);
    } finally {
      setLoading(false);
    }
  };

  const getVersionNumber = (prompt: PromptSpec) => {
    // Use explicit version field if available, otherwise try to extract from litellm_params.prompt_id
    if (prompt.version) {
      return `v${prompt.version}`;
    }

    // Fallback: try to extract from litellm_params.prompt_id
    const versionedId = (prompt.litellm_params as any)?.prompt_id || prompt.prompt_id;
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

  if (!isOpen) return null;

  return (
    <aside
      role="dialog"
      aria-modal={false}
      aria-labelledby="version-history-title"
      className="fixed inset-y-0 right-0 z-overlay flex w-[400px] max-w-full flex-col gap-4 border-l border-border bg-popover text-popover-foreground shadow-lg"
    >
      <Button type="button" variant="ghost" size="icon-sm" className="absolute top-4 right-4" onClick={onClose}>
        <XIcon />
        <span className="sr-only">Close</span>
      </Button>
      <header className="flex flex-col gap-1.5 p-4">
        <h2 id="version-history-title" className="font-medium text-foreground">
          Version History
        </h2>
      </header>
      <div className="overflow-y-auto px-4 pb-4">
        {loading ? (
          <div className="space-y-3" role="status" aria-label="Loading version history">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        ) : versions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">No version history available.</div>
        ) : (
          <div className="space-y-4">
            {versions.map((item, index) => {
              // Use version field for comparison since all items have the same prompt_id
              const itemVersionNum = item.version || parseInt(getVersionNumber(item).replace("v", ""));

              // Extract version number from activeVersionId (may have .vX suffix)
              let activeVersionNum: number | null = null;
              if (activeVersionId) {
                if (activeVersionId.includes(".v")) {
                  activeVersionNum = parseInt(activeVersionId.split(".v")[1]);
                } else if (activeVersionId.includes("_v")) {
                  activeVersionNum = parseInt(activeVersionId.split("_v")[1]);
                }
              }

              // Default to latest (first item) if no activeVersionId
              const isSelected = activeVersionNum ? itemVersionNum === activeVersionNum : index === 0;

              return (
                <button
                  type="button"
                  key={`${item.prompt_id}-v${item.version || itemVersionNum}`}
                  className={`w-full p-4 rounded-lg border cursor-pointer text-left transition-all hover:shadow-md ${
                    isSelected ? "border-primary bg-accent" : "border-border bg-background hover:border-primary"
                  }`}
                  onClick={() => onSelectVersion?.(item)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{getVersionNumber(item)}</Badge>
                      {index === 0 && <Badge>Latest</Badge>}
                    </div>
                    {isSelected && <Badge variant="secondary">Active</Badge>}
                  </div>

                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-muted-foreground font-medium">{formatDate(item.created_at)}</span>
                    <span className="text-xs text-muted-foreground">
                      {item.prompt_info?.prompt_type === "db" ? "Saved to Database" : "Config Prompt"}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
};

export default VersionHistorySidePanel;
