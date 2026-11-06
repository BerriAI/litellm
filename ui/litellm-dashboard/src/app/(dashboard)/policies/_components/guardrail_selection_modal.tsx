import { useTerminology } from "@/contexts/TerminologyContext";
import React, { useState, useEffect } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { CheckCircle2, Info } from "lucide-react";
import { formatGuardrailMode } from "@/app/(dashboard)/guardrails/_components/guardrail_info_helpers";

interface GuardrailInfo {
  guardrail_name: string;
  description: string;
  alreadyExists: boolean;
  definition: any;
}

interface GuardrailSelectionModalProps {
  visible: boolean;
  template: any;
  existingGuardrails: Set<string>;
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
  const term = useTerminology();
  const { t } = useTranslation();
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(new Set());

  // Prepare guardrail info with existence status
  const guardrailsInfo: GuardrailInfo[] = (template?.guardrailDefinitions || []).map((def: any) => ({
    guardrail_name: def.guardrail_name,
    description: def.guardrail_info?.description || t("policies.guardrailSelection.noDescription"),
    alreadyExists: existingGuardrails.has(def.guardrail_name),
    definition: def,
  }));

  // Initialize selection: select only new guardrails by default
  useEffect(() => {
    if (visible && template) {
      const newGuardrails = guardrailsInfo.filter((g) => !g.alreadyExists).map((g) => g.guardrail_name);
      setSelectedGuardrails(new Set(newGuardrails));
    }
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
    const allNew = guardrailsInfo.filter((g) => !g.alreadyExists).map((g) => g.guardrail_name);
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

  const newGuardrailsCount = guardrailsInfo.filter((g) => !g.alreadyExists).length;
  const existingCount = guardrailsInfo.filter((g) => g.alreadyExists).length;
  const selectedCount = selectedGuardrails.size;

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="sm:max-w-175">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg">
            {template?.title}
            {progressInfo && (
              <Badge variant="secondary">
                {t("policies.guardrailSelection.progress", {
                  current: progressInfo.current,
                  total: progressInfo.total,
                })}
              </Badge>
            )}
          </DialogTitle>
          <DialogDescription>{t("policies.guardrailSelection.description")}</DialogDescription>
        </DialogHeader>

        <div className="py-4">
          {/* Summary Stats */}
          <div className="mb-4 flex items-center gap-4 rounded-lg border border-border bg-muted p-3">
            <Info className="size-4 text-muted-foreground" />
            <div className="flex-1">
              <div className="text-sm">
                <span className="font-medium">
                  {t("policies.guardrailSelection.totalCount", { total: guardrailsInfo.length })}
                </span>
                <span className="mx-2 text-muted-foreground">•</span>
                <span className="font-medium text-success">
                  {t("policies.guardrailSelection.newCount", { newCount: newGuardrailsCount })}
                </span>
                {existingCount > 0 && (
                  <>
                    <span className="mx-2 text-muted-foreground">•</span>
                    <span className="text-muted-foreground">
                      {t("policies.guardrailSelection.existingCount", { existing: existingCount })}
                    </span>
                  </>
                )}
              </div>
            </div>
            {newGuardrailsCount > 0 && (
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleSelectAll}>
                  {t("policies.guardrailSelection.selectAllNew")}
                </Button>
                <Button variant="outline" size="sm" onClick={handleDeselectAll}>
                  {t("policies.guardrailSelection.deselectAll")}
                </Button>
              </div>
            )}
          </div>

          {/* Guardrails List */}
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {guardrailsInfo.map((guardrail) => (
              <div
                key={guardrail.guardrail_name}
                className={`rounded-lg border p-4 transition-colors ${
                  guardrail.alreadyExists ? "border-border bg-muted/50" : "border-border bg-card hover:border-ring"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="shrink-0 pt-0.5">
                    {guardrail.alreadyExists ? (
                      <CheckCircle2 className="size-4 text-success" />
                    ) : (
                      <Checkbox
                        checked={selectedGuardrails.has(guardrail.guardrail_name)}
                        onCheckedChange={() => handleToggle(guardrail.guardrail_name)}
                      />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-sm font-medium">{guardrail.guardrail_name}</span>
                      {guardrail.alreadyExists && (
                        <Badge variant="secondary">{t("policies.guardrailSelection.alreadyExists")}</Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{guardrail.description}</p>

                    {/* Show guardrail type and mode */}
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline">
                        {guardrail.definition?.litellm_params?.guardrail ||
                          t("policies.guardrailSelection.unknownProvider")}
                      </Badge>
                      <Badge variant="secondary">
                        {formatGuardrailMode(guardrail.definition?.litellm_params?.mode) ||
                          t("policies.guardrailSelection.unknownMode")}
                      </Badge>
                      {guardrail.definition?.litellm_params?.patterns && (
                        <Badge variant="secondary">
                          {t("policies.guardrailSelection.patterns", {
                            patterns: guardrail.definition.litellm_params.patterns.length,
                          })}
                        </Badge>
                      )}
                      {guardrail.definition?.litellm_params?.categories && (
                        <Badge variant="secondary">
                          {t("policies.guardrailSelection.categories", {
                            categories: guardrail.definition.litellm_params.categories.length,
                          })}
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {guardrailsInfo.length === 0 && (
            <div className="py-8 text-center text-muted-foreground">
              <p>{t("policies.guardrailSelection.empty.title")}</p>
              <p className="text-sm mt-2">{t("policies.guardrailSelection.empty.description")}</p>
            </div>
          )}

          {/* Discovered Competitors */}
          {template?.discoveredCompetitors?.length > 0 && (
            <>
              <Separator className="my-4" />
              <div className="rounded-lg border border-border bg-muted p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className="text-lg">✨</span>
                  <span className="text-sm font-medium">
                    {term(t("policies.guardrailSelection.discoveredCompetitors"))} (
                    {template.discoveredCompetitors.length})
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {template.discoveredCompetitors.map((name: string) => (
                    <Badge key={name} variant="secondary">
                      {name}
                    </Badge>
                  ))}
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{t("policies.guardrailSelection.competitorsNote")}</p>
              </div>
            </>
          )}

          <Separator className="my-4" />

          {/* Selected Summary */}
          <div className="text-sm text-muted-foreground">
            {selectedCount > 0 ? (
              <p>
                <Trans
                  i18nKey="policies.guardrailSelection.willBeCreated"
                  count={selectedCount}
                  components={{ highlight: <span className="font-medium text-foreground" /> }}
                />
              </p>
            ) : existingCount > 0 ? (
              <p className="text-success">{t("policies.guardrailSelection.allExist")}</p>
            ) : (
              <p className="text-warning">{t("policies.guardrailSelection.selectAtLeastOne")}</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isLoading}>
            {t("policies.guardrailSelection.cancel")}
          </Button>
          <Button onClick={handleConfirm} disabled={isLoading || (selectedCount === 0 && existingCount === 0)}>
            {selectedCount > 0
              ? t("policies.guardrailSelection.createAndUse", { count: selectedCount })
              : t("policies.guardrailSelection.useTemplate")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default GuardrailSelectionModal;
