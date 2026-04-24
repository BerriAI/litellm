import React, { useCallback, useState, useEffect, useMemo } from "react";
import { Controller, FormProvider, useForm, useFormContext } from "react-hook-form";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { AlertCircle, Info, X } from "lucide-react";
import {
  resolvePoliciesCall,
  teamListCall,
  keyListCall,
  modelAvailableCall,
} from "../networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface PolicyTestPanelProps {
  accessToken: string | null;
}

interface PolicyMatchDetail {
  policy_name: string;
  matched_via: string;
  guardrails_added: string[];
}

interface ResolveResult {
  effective_guardrails: string[];
  matched_policies: PolicyMatchDetail[];
}

interface PolicyTestFormValues {
  team_alias: string | undefined;
  key_alias: string | undefined;
  model: string | undefined;
  tags: string[];
}

function ClearableSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string | undefined;
  onChange: (v: string | undefined) => void;
  options: { label: string; value: string }[];
  placeholder: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Select value={value ?? ""} onValueChange={(v) => onChange(v || undefined)}>
        <SelectTrigger className="flex-1">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No options
            </div>
          ) : (
            options.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {value && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => onChange(undefined)}
        >
          Clear
        </Button>
      )}
    </div>
  );
}

function TagInput({
  value,
  onChange,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  placeholder: string;
}) {
  const [draft, setDraft] = useState("");
  const selected = value ?? [];

  const commit = (raw: string) => {
    const parts = raw
      .split(/[,\s]/)
      .map((p) => p.trim())
      .filter(Boolean);
    const next = [...selected];
    for (const p of parts) {
      if (!next.includes(p)) next.push(p);
    }
    onChange(next);
    setDraft("");
  };

  return (
    <div className="space-y-2">
      <Input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            if (draft.trim()) commit(draft);
          } else if (e.key === "Backspace" && !draft && selected.length > 0) {
            onChange(selected.slice(0, -1));
          }
        }}
        onBlur={() => {
          if (draft.trim()) commit(draft);
        }}
      />
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge
              key={v}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {v}
              <button
                type="button"
                onClick={() => onChange(selected.filter((s) => s !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

interface FieldsProps {
  availableTeams: string[];
  availableKeys: string[];
  availableModels: string[];
}

function PolicyTestFields({
  availableTeams,
  availableKeys,
  availableModels,
}: FieldsProps) {
  const { control } = useFormContext<PolicyTestFormValues>();

  const teamOptions = useMemo(
    () => availableTeams.map((t) => ({ label: t, value: t })),
    [availableTeams],
  );
  const keyOptions = useMemo(
    () => availableKeys.map((k) => ({ label: k, value: k })),
    [availableKeys],
  );
  const modelOptions = useMemo(
    () => availableModels.map((m) => ({ label: m, value: m })),
    [availableModels],
  );

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="space-y-2">
        <Label>Team Alias</Label>
        <Controller
          control={control}
          name="team_alias"
          render={({ field }) => (
            <ClearableSelect
              value={field.value}
              onChange={field.onChange}
              options={teamOptions}
              placeholder="Select a team alias"
            />
          )}
        />
      </div>
      <div className="space-y-2">
        <Label>Key Alias</Label>
        <Controller
          control={control}
          name="key_alias"
          render={({ field }) => (
            <ClearableSelect
              value={field.value}
              onChange={field.onChange}
              options={keyOptions}
              placeholder="Select a key alias"
            />
          )}
        />
      </div>
      <div className="space-y-2">
        <Label>Model</Label>
        <Controller
          control={control}
          name="model"
          render={({ field }) => (
            <ClearableSelect
              value={field.value}
              onChange={field.onChange}
              options={modelOptions}
              placeholder="Select a model"
            />
          )}
        />
      </div>
      <div className="space-y-2">
        <Label>Tags</Label>
        <Controller
          control={control}
          name="tags"
          render={({ field }) => (
            <TagInput
              value={field.value ?? []}
              onChange={field.onChange}
              placeholder="Type a tag and press Enter"
            />
          )}
        />
      </div>
    </div>
  );
}

const PolicyTestPanel: React.FC<PolicyTestPanelProps> = ({ accessToken }) => {
  const form = useForm<PolicyTestFormValues>({
    defaultValues: {
      team_alias: undefined,
      key_alias: undefined,
      model: undefined,
      tags: [],
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResolveResult | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const { userId, userRole } = useAuthorized();

  const loadOptions = useCallback(async () => {
    if (!accessToken) return;

    try {
      const teamsResponse = await teamListCall(accessToken, null, userId);
      const teamsArray = Array.isArray(teamsResponse)
        ? teamsResponse
        : teamsResponse?.data || [];
      setAvailableTeams(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        teamsArray.map((t: any) => t.team_alias).filter(Boolean),
      );
    } catch (error) {
      console.error("Failed to load teams:", error);
    }

    try {
      const keysResponse = await keyListCall(
        accessToken,
        null,
        null,
        null,
        null,
        null,
        1,
        100,
      );
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      setAvailableKeys(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        keysArray.map((k: any) => k.key_alias).filter(Boolean),
      );
    } catch (error) {
      console.error("Failed to load keys:", error);
    }

    try {
      const modelsResponse = await modelAvailableCall(
        accessToken,
        userId || "",
        userRole || "",
      );
      const modelsArray =
        modelsResponse?.data ||
        (Array.isArray(modelsResponse) ? modelsResponse : []);
      setAvailableModels(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        modelsArray.map((m: any) => m.id || m.model_name).filter(Boolean),
      );
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  }, [accessToken, userId, userRole]);

  useEffect(() => {
    if (accessToken) {
      loadOptions();
    }
  }, [accessToken, loadOptions]);

  const handleTest = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    setHasSearched(true);
    try {
      const values = form.getValues();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const context: any = {};
      if (values.team_alias) context.team_alias = values.team_alias;
      if (values.key_alias) context.key_alias = values.key_alias;
      if (values.model) context.model = values.model;
      if (values.tags && values.tags.length > 0) context.tags = values.tags;

      const data = await resolvePoliciesCall(accessToken, context);
      setResult(data);
    } catch (error) {
      console.error("Error resolving policies:", error);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    form.reset({
      team_alias: undefined,
      key_alias: undefined,
      model: undefined,
      tags: [],
    });
    setResult(null);
    setHasSearched(false);
  };

  return (
    <div>
      <div className="bg-background border border-border rounded-lg p-6 mb-6">
        <div className="mb-5">
          <h3 className="text-base font-semibold mb-1">Policy Simulator</h3>
          <p className="text-muted-foreground text-sm">
            Simulate a request to see which policies and guardrails would
            apply. Select a team, key, model, or tags below and click
            &quot;Simulate&quot; to see the results.
          </p>
        </div>

        <FormProvider {...form}>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleTest();
            }}
          >
            <PolicyTestFields
              availableTeams={availableTeams}
              availableKeys={availableKeys}
              availableModels={availableModels}
            />
            <div className="flex space-x-2 mt-4">
              <Button
                type="submit"
                disabled={!accessToken || isLoading}
              >
                {isLoading ? "Simulating..." : "Simulate"}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={handleReset}
              >
                Reset
              </Button>
            </div>
          </form>
        </FormProvider>
      </div>

      {!hasSearched && (
        <div className="bg-background border border-border rounded-lg p-8 text-center">
          <div className="text-muted-foreground/80 mb-2">
            <Info className="h-10 w-10 mx-auto mb-3" />
          </div>
          <p className="text-sm font-medium text-muted-foreground mb-1">
            No simulation run yet
          </p>
          <p className="text-xs text-muted-foreground/80">
            Fill in one or more fields above and click &quot;Simulate&quot; to
            see which policies and guardrails would apply to that request.
          </p>
        </div>
      )}

      {hasSearched && result && (
        <div className="bg-background border border-border rounded-lg p-6">
          {result.matched_policies.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <p>No policies matched this context</p>
            </div>
          ) : (
            <>
              <div className="mb-4">
                <p className="text-sm font-semibold mb-2">
                  Effective Guardrails
                </p>
                <div className="flex flex-wrap gap-1">
                  {result.effective_guardrails.length > 0 ? (
                    result.effective_guardrails.map((g) => (
                      <Badge
                        key={g}
                        className="bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                      >
                        {g}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-muted-foreground text-sm">None</span>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Matched Policies</p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Policy</th>
                      <th className="text-left py-2 pr-4">Matched Via</th>
                      <th className="text-left py-2">Guardrails Added</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.matched_policies.map((p) => (
                      <tr
                        key={p.policy_name}
                        className="border-b border-border last:border-0"
                      >
                        <td className="py-2 pr-4 font-medium">
                          {p.policy_name}
                        </td>
                        <td className="py-2 pr-4">
                          <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                            {p.matched_via}
                          </Badge>
                        </td>
                        <td className="py-2">
                          {p.guardrails_added.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {p.guardrails_added.map((g) => (
                                <Badge
                                  key={g}
                                  className="bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                                >
                                  {g}
                                </Badge>
                              ))}
                            </div>
                          ) : (
                            <span className="text-muted-foreground">None</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {hasSearched && !result && !isLoading && (
        <div
          className={cn(
            "flex gap-2 items-start p-3 rounded-md bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 text-red-800 dark:text-red-200",
          )}
        >
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold">Error</div>
            <div className="text-sm">
              Failed to resolve policies. Check the proxy logs.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PolicyTestPanel;
