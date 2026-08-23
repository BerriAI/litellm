"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/shared/form/field";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import CopyButton from "@/components/shared/CopyButton";
import { AlertTriangle, ArrowLeft, ArrowRight, Loader2 } from "lucide-react";
import { Providers } from "@/components/provider_info_helpers";
import type { AnthropicJwks } from "@/components/networking";
import type { CreationResult } from "./wizardLogic";

const CREATION_RESULT_CLASS_NAME: Record<CreationResult["status"], string> = {
  failed: "text-destructive",
  skipped: "text-muted-foreground",
  created: "text-success",
};

interface ProviderStepProps {
  providerOptions: SearchSelectOption[];
  selectedProvider: Providers | null;
  onSelectProvider: (provider: Providers) => void;
  credentialName: string;
  onCredentialNameChange: (name: string) => void;
  nameCollision: boolean;
  onNext: () => void;
}

export const ProviderStep: React.FC<ProviderStepProps> = ({
  providerOptions,
  selectedProvider,
  onSelectProvider,
  credentialName,
  onCredentialNameChange,
  nameCollision,
  onNext,
}) => (
  <Card>
    <CardContent className="space-y-4">
      <Field>
        <FieldLabel htmlFor="add-provider-provider">Provider</FieldLabel>
        <SearchSelect
          inputId="add-provider-provider"
          options={providerOptions}
          placeholder="Select a provider"
          value={selectedProvider ?? ""}
          onValueChange={(value) => onSelectProvider(value as Providers)}
        />
      </Field>
      <Field>
        <FieldLabel htmlFor="add-provider-credential-name">Credential name</FieldLabel>
        <Input
          id="add-provider-credential-name"
          value={credentialName}
          onChange={(e) => onCredentialNameChange(e.target.value)}
          placeholder="e.g. anthropic-prod"
        />
        {nameCollision && <p className="text-sm text-destructive">A credential with this name already exists.</p>}
      </Field>
      <div className="flex justify-end">
        <Button disabled={!selectedProvider || !credentialName || nameCollision} onClick={onNext}>
          Next <ArrowRight className="ml-1 size-4" />
        </Button>
      </div>
    </CardContent>
  </Card>
);

interface JwksStepProps {
  jwks: AnthropicJwks | null;
  jwksError: string | null;
  federationRuleId: string;
  onFederationRuleIdChange: (value: string) => void;
  onBack: () => void;
  onNext: () => void;
}

export const JwksStep: React.FC<JwksStepProps> = ({
  jwks,
  jwksError,
  federationRuleId,
  onFederationRuleIdChange,
  onBack,
  onNext,
}) => (
  <Card>
    <CardContent className="space-y-4">
      <Alert variant="info">
        <AlertTitle>Register this JWKS with Anthropic</AlertTitle>
        <AlertDescription>
          Register this public JWKS as the inline issuer for your federation rule in the Anthropic Console, then
          paste the resulting Federation Rule ID below.
        </AlertDescription>
      </Alert>
      {jwksError && (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>Could not load JWKS</AlertTitle>
          <AlertDescription>{jwksError}</AlertDescription>
        </Alert>
      )}
      {jwks && (
        <div className="relative rounded-md border bg-muted p-3">
          <CopyButton value={JSON.stringify(jwks, null, 2)} label="Copy JWKS" className="absolute top-2 right-2" />
          <pre className="overflow-x-auto text-xs">{JSON.stringify(jwks, null, 2)}</pre>
        </div>
      )}
      <Field>
        <FieldLabel htmlFor="add-provider-federation-rule-id">Federation Rule ID</FieldLabel>
        <Input
          id="add-provider-federation-rule-id"
          value={federationRuleId}
          onChange={(e) => onFederationRuleIdChange(e.target.value)}
        />
      </Field>
      <div className="flex justify-between">
        <Button type="button" variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-1 size-4" /> Back
        </Button>
        <Button disabled={!federationRuleId} onClick={onNext}>
          Next <ArrowRight className="ml-1 size-4" />
        </Button>
      </div>
    </CardContent>
  </Card>
);

interface DiscoverStepProps {
  isDiscovering: boolean;
  discoveryError: string | null;
  onBack: () => void;
  onRetry: () => void;
}

export const DiscoverStep: React.FC<DiscoverStepProps> = ({ isDiscovering, discoveryError, onBack, onRetry }) => (
  <Card>
    <CardContent className="space-y-4">
      {isDiscovering && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Discovering models...
        </p>
      )}
      {discoveryError && (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>Discovery failed</AlertTitle>
          <AlertDescription>{discoveryError}</AlertDescription>
        </Alert>
      )}
      <div className="flex justify-between">
        <Button type="button" variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-1 size-4" /> Back
        </Button>
        {discoveryError && (
          <Button disabled={isDiscovering} onClick={onRetry}>
            Retry
          </Button>
        )}
      </div>
    </CardContent>
  </Card>
);

interface ResultsStepProps {
  isCreating: boolean;
  isDone: boolean;
  creationResults: CreationResult[];
  aliasCollisions: string[];
}

export const ResultsStep: React.FC<ResultsStepProps> = ({ isCreating, isDone, creationResults, aliasCollisions }) => (
  <Card>
    <CardContent className="space-y-4">
      {isCreating && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Creating models...
        </p>
      )}
      {isDone && (
        <>
          <ul className="space-y-1 text-sm">
            {creationResults.map((result) => (
              <li key={result.row.id}>
                <span className={CREATION_RESULT_CLASS_NAME[result.status]}>
                  {result.row.modelName}: {result.status}
                  {result.detail ? ` (${result.detail})` : ""}
                </span>
              </li>
            ))}
          </ul>
          {aliasCollisions.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="size-4" />
              <AlertTitle>Some alternate names were not saved</AlertTitle>
              <AlertDescription>
                These alias names already exist and were left unchanged: {aliasCollisions.join(", ")}
              </AlertDescription>
            </Alert>
          )}
        </>
      )}
    </CardContent>
  </Card>
);
