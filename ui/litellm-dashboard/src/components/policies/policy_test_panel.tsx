import React, { useCallback, useState, useEffect } from "react";
import { Form, Select } from "antd";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AlertCircle, Info } from "lucide-react";
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

const PolicyTestPanel: React.FC<PolicyTestPanelProps> = ({ accessToken }) => {
  const [form] = Form.useForm();
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
      const values = form.getFieldsValue(true);
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
    form.resetFields();
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

        <Form form={form} layout="vertical">
          <div className="grid grid-cols-2 gap-4">
            <Form.Item name="team_alias" label="Team Alias" className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder="Select or type a team alias"
                options={availableTeams.map((t) => ({ label: t, value: t }))}
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>
            <Form.Item name="key_alias" label="Key Alias" className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder="Select or type a key alias"
                options={availableKeys.map((k) => ({ label: k, value: k }))}
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>
            <Form.Item name="model" label="Model" className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder="Select or type a model"
                options={availableModels.map((m) => ({ label: m, value: m }))}
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
              />
            </Form.Item>
            <Form.Item name="tags" label="Tags" className="mb-3">
              <Select
                mode="tags"
                placeholder="Type a tag and press Enter"
                tokenSeparators={[",", " "]}
                notFoundContent={null}
                suffixIcon={null}
                open={false}
              />
            </Form.Item>
          </div>
          <div className="flex space-x-2">
            <Button
              onClick={handleTest}
              disabled={!accessToken || isLoading}
            >
              {isLoading ? "Simulating..." : "Simulate"}
            </Button>
            <Button variant="secondary" onClick={handleReset}>
              Reset
            </Button>
          </div>
        </Form>
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
