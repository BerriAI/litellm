"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Shrink } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { createGuardrailCall, getGuardrailsList } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  buildCompressionGuardrailPayload,
  compressionGuardrailsOf,
  GuardrailListItem,
  GuardrailListResponse,
} from "./helpers";

interface CompressionViewProps {
  accessToken: string | null;
}

interface CompressionFormValues {
  name: string;
  apiBase: string;
  defaultOn: boolean;
}

const EMPTY_FORM: CompressionFormValues = { name: "", apiBase: "", defaultOn: true };

const CompressionView: React.FC<CompressionViewProps> = ({ accessToken }) => {
  const [formValues, setFormValues] = useState<CompressionFormValues>(EMPTY_FORM);
  const [guardrails, setGuardrails] = useState<GuardrailListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadGuardrails = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGuardrailsList(accessToken)
      .then((response) => setGuardrails(compressionGuardrailsOf(response as GuardrailListResponse)))
      .catch((error) => {
        console.error("Failed to load compression guardrails:", error);
        NotificationsManager.fromBackend("Failed to load compression guardrails");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken]);

  useEffect(() => {
    loadGuardrails();
  }, [loadGuardrails]);

  const handleAdd = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!accessToken) {
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(accessToken, buildCompressionGuardrailPayload(formValues));
      NotificationsManager.success("Compression guardrail created");
      setFormValues(EMPTY_FORM);
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to create compression guardrail:", error);
      NotificationsManager.fromBackend("Failed to create compression guardrail");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6 p-6">
      <div>
        <div className="flex items-center gap-2">
          <Shrink className="size-6 text-emerald-600" strokeWidth={1.75} />
          <h1 className="text-xl font-semibold text-foreground">Compression</h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure prompt compression so you pay for fewer input tokens
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Headroom prompt compression</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            Headroom is a native LiteLLM guardrail that compresses your prompts before they reach the model, so you pay
            for fewer input tokens. The tokens it removes are priced and shown as compression savings on the Cost
            Optimization dashboard.{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/headroom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline"
            >
              Headroom setup docs
            </a>
          </p>
          {isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
          {!isLoading && guardrails.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No prompt compression guardrails configured yet. Add one below to start saving on input tokens
            </p>
          )}
          {!isLoading && guardrails.length > 0 && (
            <ul className="divide-y divide-gray-200">
              {guardrails.map((guardrail) => (
                <li key={guardrail.guardrail_id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">{guardrail.guardrail_name}</p>
                    <p className="text-xs text-muted-foreground">{guardrail.litellm_params?.api_base ?? ""}</p>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      guardrail.litellm_params?.default_on
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {guardrail.litellm_params?.default_on ? "Always on" : "Opt-in"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Add Headroom compression guardrail</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleAdd} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="compression-name">Name</Label>
              <Input
                id="compression-name"
                required
                placeholder="headroom-compression"
                value={formValues.name}
                onChange={(event) => setFormValues({ ...formValues, name: event.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="compression-api-base">Headroom API base</Label>
              <Input
                id="compression-api-base"
                required
                placeholder="https://your-headroom-endpoint"
                value={formValues.apiBase}
                onChange={(event) => setFormValues({ ...formValues, apiBase: event.target.value })}
              />
              <p className="text-xs text-muted-foreground">
                Base URL of your Headroom compression service; LiteLLM calls its /v1/compress endpoint
              </p>
            </div>
            <Label htmlFor="compression-default-on">
              <Switch
                id="compression-default-on"
                checked={formValues.defaultOn}
                onCheckedChange={(checked) => setFormValues({ ...formValues, defaultOn: checked })}
              />
              Apply to all requests
            </Label>
            <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3">
              <p className="text-sm text-yellow-800">
                Applying compression to all requests is available to all users. Enabling it selectively per key or team
                is a LiteLLM Enterprise feature. Get a trial key{" "}
                <a
                  href="https://www.litellm.ai/#pricing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  here
                </a>
              </p>
            </div>
            <div className="flex justify-end">
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "Adding..." : "Add guardrail"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default CompressionView;
