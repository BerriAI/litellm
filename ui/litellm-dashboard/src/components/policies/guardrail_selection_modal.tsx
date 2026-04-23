import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CheckCircle2, Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface GuardrailInfo {
  guardrail_name: string;
  description: string;
  alreadyExists: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  definition: any;
}

interface GuardrailSelectionModalProps {
  visible: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  template: any;
  existingGuardrails: Set<string>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onConfirm: (selectedGuardrails: any[]) => void;
  onCancel: () => void;
  isLoading?: boolean;
  progressInfo?: { current: number; total: number } | null;
}

const GuardrailSelectionModal: React.FC<GuardrailSelectionModalProps> = ({
  visible,
  template,
  existingGuardrails,
  onConfirm,
  onCancel,
  isLoading = false,
  progressInfo,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(
    new Set(),
  );

  const guardrailsInfo: GuardrailInfo[] = (
    template?.guardrailDefinitions || []
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ).map((def: any) => ({
    guardrail_name: def.guardrail_name,
    description: def.guardrail_info?.description || "No description available",
    alreadyExists: existingGuardrails.has(def.guardrail_name),
    definition: def,
  }));

  useEffect(() => {
    if (visible && template) {
      const newGuardrails = guardrailsInfo
        .filter((g) => !g.alreadyExists)
        .map((g) => g.guardrail_name);
      setSelectedGuardrails(new Set(newGuardrails));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, template]);

  const handleToggle = (guardrailName: string) => {
    setSelectedGuardrails((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(guardrailName)) {
        newSet.delete(guardrailName);
      } else {
        newSet.add(guardrailName);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    const allNew = guardrailsInfo
      .filter((g) => !g.alreadyExists)
      .map((g) => g.guardrail_name);
    setSelectedGuardrails(new Set(allNew));
  };

  const handleDeselectAll = () => {
    setSelectedGuardrails(new Set());
  };

  const handleConfirm = () => {
    const selectedDefinitions = guardrailsInfo
      .filter((g) => selectedGuardrails.has(g.guardrail_name))
      .map((g) => g.definition);
    onConfirm(selectedDefinitions);
  };

  const newGuardrailsCount = guardrailsInfo.filter(
    (g) => !g.alreadyExists,
  ).length;
  const existingCount = guardrailsInfo.filter((g) => g.alreadyExists).length;
  const selectedCount = selectedGuardrails.size;

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-[700px]">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>{template?.title}</DialogTitle>
            {progressInfo && (
              // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-600 border border-blue-100 dark:bg-blue-950/30 dark:text-blue-200 dark:border-blue-900">
                Template {progressInfo.current} of {progressInfo.total}
              </span>
            )}
          </div>
          <DialogDescription>
            Review and select guardrails to create for this template
          </DialogDescription>
        </DialogHeader>

        <div className="py-2">
          {/* Summary Stats */}
          {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
          <div className="flex items-center gap-4 mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100 dark:bg-blue-950/30 dark:border-blue-900">
            {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-300 shrink-0" />
            <div className="flex-1">
              <div className="text-sm">
                <span className="font-medium">
                  {guardrailsInfo.length} total guardrails
                </span>
                <span className="text-muted-foreground mx-2">•</span>
                {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                <span className="text-emerald-600 font-medium">
                  {newGuardrailsCount} new
                </span>
                {existingCount > 0 && (
                  <>
                    <span className="text-muted-foreground mx-2">•</span>
                    <span className="text-muted-foreground">
                      {existingCount} already exist
                    </span>
                  </>
                )}
              </div>
            </div>
            {newGuardrailsCount > 0 && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSelectAll}
                >
                  Select All New
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeselectAll}
                >
                  Deselect All
                </Button>
              </div>
            )}
          </div>

          {/* Guardrails List */}
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {guardrailsInfo.map((guardrail) => (
              <div
                key={guardrail.guardrail_name}
                className={cn(
                  "border rounded-lg p-4 transition-colors",
                  guardrail.alreadyExists
                    ? "bg-muted border-border"
                    : "bg-background border-border hover:border-primary/50",
                )}
              >
                <div className="flex items-start gap-3">
                  <div className="shrink-0 pt-0.5">
                    {guardrail.alreadyExists ? (
                      // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
                      <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                    ) : (
                      <Checkbox
                        checked={selectedGuardrails.has(
                          guardrail.guardrail_name,
                        )}
                        onCheckedChange={() =>
                          handleToggle(guardrail.guardrail_name)
                        }
                      />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-sm font-medium">
                        {guardrail.guardrail_name}
                      </span>
                      {guardrail.alreadyExists && (
                        <Badge variant="outline" className="text-xs">
                          Already exists
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {guardrail.description}
                    </p>

                    {/* Show guardrail type and mode */}
                    <div className="flex gap-2 mt-2 flex-wrap">
                      <Badge variant="outline" className="text-xs">
                        {guardrail.definition?.litellm_params?.guardrail ||
                          "unknown"}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {guardrail.definition?.litellm_params?.mode ||
                          "unknown"}
                      </Badge>
                      {guardrail.definition?.litellm_params?.patterns && (
                        <Badge variant="secondary" className="text-xs">
                          {
                            guardrail.definition.litellm_params.patterns
                              .length
                          }{" "}
                          pattern(s)
                        </Badge>
                      )}
                      {guardrail.definition?.litellm_params?.categories && (
                        <Badge variant="secondary" className="text-xs">
                          {
                            guardrail.definition.litellm_params.categories
                              .length
                          }{" "}
                          category/categories
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {guardrailsInfo.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <p>No guardrails defined for this template.</p>
              <p className="text-sm mt-2">
                This template will use existing guardrails in your system.
              </p>
            </div>
          )}

          {/* Discovered Competitors */}
          {template?.discoveredCompetitors?.length > 0 && (
            <>
              <hr className="border-border my-4" />
              {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
              <div className="p-3 bg-purple-50 rounded-lg border border-purple-100 dark:bg-purple-950/30 dark:border-purple-900">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">✨</span>
                  {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                  <span className="font-medium text-purple-900 dark:text-purple-200 text-sm">
                    AI-Discovered Competitors (
                    {template.discoveredCompetitors.length})
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {template.discoveredCompetitors.map((name: string) => (
                    <Badge key={name} variant="secondary" className="text-xs">
                      {name}
                    </Badge>
                  ))}
                </div>
                {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                <p className="text-xs text-purple-600 dark:text-purple-300 mt-2">
                  These competitor names will be automatically blocked by the
                  competitor-name-blocker guardrail.
                </p>
              </div>
            </>
          )}

          <hr className="border-border my-4" />

          {/* Selected Summary */}
          <div className="text-sm text-muted-foreground">
            {selectedCount > 0 ? (
              <p>
                <span className="font-medium text-foreground">
                  {selectedCount}
                </span>{" "}
                guardrail{selectedCount > 1 ? "s" : ""} will be created
              </p>
            ) : existingCount > 0 ? (
              // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
              <p className="text-emerald-600">
                All guardrails already exist. You can proceed to use this
                template.
              </p>
            ) : (
              // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
              <p className="text-orange-600">
                Select at least one guardrail to create, or click &quot;Use
                Template&quot; to proceed without creating new guardrails.
              </p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={
              isLoading || (selectedCount === 0 && existingCount === 0)
            }
          >
            {isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            {selectedCount > 0
              ? `Create ${selectedCount} Guardrail${
                  selectedCount > 1 ? "s" : ""
                } & Use Template`
              : "Use Template"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default GuardrailSelectionModal;
