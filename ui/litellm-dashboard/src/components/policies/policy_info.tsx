import React, { useState, useEffect, useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ArrowLeft, Info, Loader2, Pencil } from "lucide-react";
import { Policy } from "./types";
import { PipelineInfoDisplay } from "./pipeline_flow_builder";
import { getResolvedGuardrails } from "../networking";

interface PolicyInfoViewProps {
  policyId: string;
  onClose: () => void;
  onEdit: (policy: Policy) => void;
  accessToken: string | null;
  isAdmin: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getPolicy: (accessToken: string, policyId: string) => Promise<any>;
}

const Section: React.FC<{
  title: string;
  children: React.ReactNode;
}> = ({ title, children }) => (
  <div>
    <div className="flex items-center gap-2 mb-3">
      <span className="font-bold">{title}</span>
      <hr className="flex-1 border-border" />
    </div>
    {children}
  </div>
);

const InfoAlert: React.FC<{
  title: React.ReactNode;
  description?: React.ReactNode;
}> = ({ title, description }) => (
  <div className="flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200 mb-4">
    <Info className="h-4 w-4 mt-0.5 shrink-0" />
    <div className="flex-1">
      <div className="font-semibold">{title}</div>
      {description && <div className="text-sm mt-1">{description}</div>}
    </div>
  </div>
);

const DescRow: React.FC<{
  label: string;
  children: React.ReactNode;
  first?: boolean;
}> = ({ label, children, first }) => (
  <div
    className={cn(
      "grid grid-cols-[minmax(160px,220px)_1fr]",
      !first && "border-t border-border",
    )}
  >
    <div className="bg-muted px-4 py-2.5 font-medium">{label}</div>
    <div className="px-4 py-2.5">{children}</div>
  </div>
);

const PolicyInfoView: React.FC<PolicyInfoViewProps> = ({
  policyId,
  onClose,
  onEdit,
  accessToken,
  isAdmin,
  getPolicy,
}) => {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [resolvedGuardrails, setResolvedGuardrails] = useState<string[]>([]);

  const fetchPolicy = useCallback(async () => {
    if (!accessToken || !policyId) return;

    setIsLoading(true);
    try {
      const data = await getPolicy(accessToken, policyId);
      setPolicy(data);

      try {
        const resolvedData = await getResolvedGuardrails(accessToken, policyId);
        setResolvedGuardrails(resolvedData.resolved_guardrails || []);
      } catch (error) {
        console.error("Error fetching resolved guardrails:", error);
      }
    } catch (error) {
      console.error("Error fetching policy:", error);
    } finally {
      setIsLoading(false);
    }
  }, [policyId, accessToken, getPolicy]);

  useEffect(() => {
    fetchPolicy();
  }, [fetchPolicy]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center p-12">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!policy) {
    return (
      <Card className="p-4">
        <p className="text-destructive">Policy not found</p>
        <Button onClick={onClose} className="mt-4" variant="secondary">
          Go Back
        </Button>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Button variant="secondary" onClick={onClose}>
            <ArrowLeft className="h-4 w-4" />
            Back to Policies
          </Button>
          {isAdmin && (
            <Button onClick={() => onEdit(policy)}>
              <Pencil className="h-4 w-4" />
              Edit Policy
            </Button>
          )}
        </div>

        <h4 className="text-xl font-semibold">{policy.policy_name}</h4>

        <div className="border border-border rounded-md overflow-hidden">
          <DescRow label="Policy ID" first>
            <code className="text-xs bg-muted px-2 py-1 rounded">
              {policy.policy_id}
            </code>
          </DescRow>
          <DescRow label="Description">
            {policy.description || (
              <span className="text-muted-foreground">No description</span>
            )}
          </DescRow>
          <DescRow label="Inherits From">
            {policy.inherit ? (
              <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                {policy.inherit}
              </Badge>
            ) : (
              <span className="text-muted-foreground">None</span>
            )}
          </DescRow>
          <DescRow label="Created At">
            {policy.created_at
              ? new Date(policy.created_at).toLocaleString()
              : "-"}
          </DescRow>
          <DescRow label="Updated At">
            {policy.updated_at
              ? new Date(policy.updated_at).toLocaleString()
              : "-"}
          </DescRow>
        </div>

        {policy.pipeline && (
          <>
            <Section title="Pipeline Flow">
              <InfoAlert
                title={`Pipeline (${policy.pipeline.mode} mode, ${policy.pipeline.steps.length} step${policy.pipeline.steps.length !== 1 ? "s" : ""})`}
              />
              <PipelineInfoDisplay pipeline={policy.pipeline} />
            </Section>
          </>
        )}

        <Section title="Guardrails Configuration">
          {resolvedGuardrails.length > 0 && (
            <InfoAlert
              title="Resolved Guardrails"
              description={
                <div>
                  <div className="text-muted-foreground mb-2">
                    Final guardrails that will be applied (including
                    inheritance):
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {resolvedGuardrails.map((g) => (
                      <Badge
                        key={g}
                        className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                      >
                        {g}
                      </Badge>
                    ))}
                  </div>
                </div>
              }
            />
          )}

          <div className="border border-border rounded-md overflow-hidden">
            <DescRow label="Guardrails to Add" first>
              <div className="flex flex-wrap gap-1">
                {policy.guardrails_add && policy.guardrails_add.length > 0 ? (
                  policy.guardrails_add.map((g) => (
                    <Badge
                      key={g}
                      className="bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                    >
                      {g}
                    </Badge>
                  ))
                ) : (
                  <span className="text-muted-foreground">None</span>
                )}
              </div>
            </DescRow>
            <DescRow label="Guardrails to Remove">
              <div className="flex flex-wrap gap-1">
                {policy.guardrails_remove &&
                policy.guardrails_remove.length > 0 ? (
                  policy.guardrails_remove.map((g) => (
                    <Badge
                      key={g}
                      className="bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                    >
                      {g}
                    </Badge>
                  ))
                ) : (
                  <span className="text-muted-foreground">None</span>
                )}
              </div>
            </DescRow>
          </div>
        </Section>

        <Section title="Conditions">
          <div className="border border-border rounded-md overflow-hidden">
            <DescRow label="Model Condition" first>
              {policy.condition?.model ? (
                <Badge className="bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300">
                  {typeof policy.condition.model === "string"
                    ? policy.condition.model
                    : JSON.stringify(policy.condition.model)}
                </Badge>
              ) : (
                <span className="text-muted-foreground">
                  No model condition (applies to all models)
                </span>
              )}
            </DescRow>
          </div>
        </Section>
      </div>
    </Card>
  );
};

export default PolicyInfoView;
