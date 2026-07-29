import React, { useState, useEffect, useCallback } from "react";
import { ArrowLeft, Info, Pencil } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Policy } from "@/components/policies/types";
import { PipelineInfoDisplay } from "./pipeline_flow_builder";
import { getResolvedGuardrails } from "@/components/networking";

interface PolicyInfoViewProps {
  policyId: string;
  onClose: () => void;
  onEdit: (policy: Policy) => void;
  accessToken: string | null;
  isAdmin: boolean;
  getPolicy: (accessToken: string, policyId: string) => Promise<any>;
}

interface DetailRowProps {
  label: string;
  children: React.ReactNode;
}

const DetailRow = ({ label, children }: DetailRowProps) => (
  <div className="grid grid-cols-1 border-b border-border last:border-b-0 sm:grid-cols-[200px_minmax(0,1fr)]">
    <dt className="bg-muted/50 px-4 py-3 text-sm font-medium">{label}</dt>
    <dd className="px-4 py-3 text-sm">{children}</dd>
  </div>
);

const SectionHeading = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-center gap-3">
    <span className="text-sm font-semibold">{children}</span>
    <Separator className="flex-1" />
  </div>
);

const Muted = ({ children }: { children: React.ReactNode }) => (
  <span className="text-muted-foreground">{children}</span>
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
  const [isLoadingResolved, setIsLoadingResolved] = useState(false);

  const fetchPolicy = useCallback(async () => {
    if (!accessToken || !policyId) return;

    setIsLoading(true);
    try {
      const data = await getPolicy(accessToken, policyId);
      setPolicy(data);

      // Also fetch resolved guardrails
      setIsLoadingResolved(true);
      try {
        const resolvedData = await getResolvedGuardrails(accessToken, policyId);
        setResolvedGuardrails(resolvedData.resolved_guardrails || []);
      } catch (error) {
        console.error("Error fetching resolved guardrails:", error);
      } finally {
        setIsLoadingResolved(false);
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
      <div className="flex flex-col items-center gap-3 p-12">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full max-w-2xl" />
      </div>
    );
  }

  if (!policy) {
    return (
      <Card>
        <CardContent>
          <p className="text-destructive">Policy not found</p>
          <Button variant="secondary" onClick={onClose} className="mt-4">
            Go Back
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <Button variant="secondary" onClick={onClose}>
              <ArrowLeft />
              Back to Policies
            </Button>
            {isAdmin && (
              <Button onClick={() => onEdit(policy)}>
                <Pencil />
                Edit Policy
              </Button>
            )}
          </div>

          <h4 className="text-lg font-semibold">{policy.policy_name}</h4>

          <dl className="rounded-md border border-border">
            <DetailRow label="Policy ID">
              <code className="rounded-sm bg-muted px-2 py-1 text-xs">{policy.policy_id}</code>
            </DetailRow>
            <DetailRow label="Description">{policy.description || <Muted>No description</Muted>}</DetailRow>
            <DetailRow label="Inherits From">
              {policy.inherit ? <Badge variant="secondary">{policy.inherit}</Badge> : <Muted>None</Muted>}
            </DetailRow>
            <DetailRow label="Created At">
              {policy.created_at ? new Date(policy.created_at).toLocaleString() : "-"}
            </DetailRow>
            <DetailRow label="Updated At">
              {policy.updated_at ? new Date(policy.updated_at).toLocaleString() : "-"}
            </DetailRow>
          </dl>

          {policy.pipeline && (
            <>
              <SectionHeading>Pipeline Flow</SectionHeading>
              <Alert className="mb-4">
                <Info />
                <AlertTitle>
                  Pipeline ({policy.pipeline.mode} mode, {policy.pipeline.steps.length} step
                  {policy.pipeline.steps.length !== 1 ? "s" : ""})
                </AlertTitle>
              </Alert>
              <PipelineInfoDisplay pipeline={policy.pipeline} />
            </>
          )}

          <SectionHeading>Guardrails Configuration</SectionHeading>

          {resolvedGuardrails.length > 0 && (
            <Alert className="mb-4">
              <Info />
              <AlertTitle>Resolved Guardrails</AlertTitle>
              <AlertDescription>
                <span className="mb-2 block">Final guardrails that will be applied (including inheritance):</span>
                <div className="flex flex-wrap gap-1">
                  {resolvedGuardrails.map((g) => (
                    <Badge key={g} variant="secondary">
                      {g}
                    </Badge>
                  ))}
                </div>
              </AlertDescription>
            </Alert>
          )}

          <dl className="rounded-md border border-border">
            <DetailRow label="Guardrails to Add">
              <div className="flex flex-wrap gap-1">
                {policy.guardrails_add && policy.guardrails_add.length > 0 ? (
                  policy.guardrails_add.map((g) => (
                    <Badge key={g} variant="secondary">
                      {g}
                    </Badge>
                  ))
                ) : (
                  <Muted>None</Muted>
                )}
              </div>
            </DetailRow>
            <DetailRow label="Guardrails to Remove">
              <div className="flex flex-wrap gap-1">
                {policy.guardrails_remove && policy.guardrails_remove.length > 0 ? (
                  policy.guardrails_remove.map((g) => (
                    <Badge key={g} variant="destructive">
                      {g}
                    </Badge>
                  ))
                ) : (
                  <Muted>None</Muted>
                )}
              </div>
            </DetailRow>
          </dl>

          <SectionHeading>Conditions</SectionHeading>

          <dl className="rounded-md border border-border">
            <DetailRow label="Model Condition">
              {policy.condition?.model ? (
                <Badge variant="secondary">
                  {typeof policy.condition.model === "string"
                    ? policy.condition.model
                    : JSON.stringify(policy.condition.model)}
                </Badge>
              ) : (
                <Muted>No model condition (applies to all models)</Muted>
              )}
            </DetailRow>
          </dl>
        </div>
      </CardContent>
    </Card>
  );
};

export default PolicyInfoView;
