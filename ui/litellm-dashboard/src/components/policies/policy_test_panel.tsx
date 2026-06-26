import React, { useState, useEffect } from "react";
import { Form, Select, Alert, Tag, Empty, Typography } from "antd";
import { Button } from "@tremor/react";
import { useTranslation } from "react-i18next";
import { resolvePoliciesCall, teamListCall, keyListCall, modelAvailableCall } from "../networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const { Text } = Typography;

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
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResolveResult | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const { userId, userRole } = useAuthorized();

  useEffect(() => {
    if (accessToken) {
      loadOptions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const loadOptions = async () => {
    if (!accessToken) return;

    try {
      const teamsResponse = await teamListCall(accessToken, null, userId);
      const teamsArray = Array.isArray(teamsResponse) ? teamsResponse : teamsResponse?.data || [];
      setAvailableTeams(teamsArray.map((t: any) => t.team_alias).filter(Boolean));
    } catch (error) {
      console.error("Failed to load teams:", error);
    }

    try {
      const keysResponse = await keyListCall(accessToken, null, null, null, null, null, 1, 100);
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      setAvailableKeys(keysArray.map((k: any) => k.key_alias).filter(Boolean));
    } catch (error) {
      console.error("Failed to load keys:", error);
    }

    try {
      const modelsResponse = await modelAvailableCall(accessToken, userId || "", userRole || "");
      const modelsArray = modelsResponse?.data || (Array.isArray(modelsResponse) ? modelsResponse : []);
      setAvailableModels(modelsArray.map((m: any) => m.id || m.model_name).filter(Boolean));
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  };

  const handleTest = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    setHasSearched(true);
    try {
      const values = form.getFieldsValue(true);
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
      <div className="bg-white border rounded-lg p-6 mb-6">
        <div className="mb-5">
          <h3 className="text-base font-semibold mb-1">{t("policies.policyTestPanel.title")}</h3>
          <Text type="secondary">{t("policies.policyTestPanel.subtitle")}</Text>
        </div>

        <Form form={form} layout="vertical">
          <div className="grid grid-cols-2 gap-4">
            <Form.Item name="team_alias" label={t("policies.policyTestPanel.teamAliasLabel")} className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder={t("policies.policyTestPanel.teamAliasPlaceholder")}
                options={availableTeams.map((team) => ({ label: team, value: team }))}
                filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              />
            </Form.Item>
            <Form.Item name="key_alias" label={t("policies.policyTestPanel.keyAliasLabel")} className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder={t("policies.policyTestPanel.keyAliasPlaceholder")}
                options={availableKeys.map((k) => ({ label: k, value: k }))}
                filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              />
            </Form.Item>
            <Form.Item name="model" label={t("policies.policyTestPanel.modelLabel")} className="mb-3">
              <Select
                showSearch
                allowClear
                placeholder={t("policies.policyTestPanel.modelPlaceholder")}
                options={availableModels.map((m) => ({ label: m, value: m }))}
                filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              />
            </Form.Item>
            <Form.Item name="tags" label={t("policies.policyTestPanel.tagsLabel")} className="mb-3">
              <Select
                mode="tags"
                placeholder={t("policies.policyTestPanel.tagsPlaceholder")}
                tokenSeparators={[",", " "]}
                notFoundContent={null}
                suffixIcon={null}
                open={false}
              />
            </Form.Item>
          </div>
          <div className="flex space-x-2">
            <Button onClick={handleTest} loading={isLoading} disabled={!accessToken}>
              {t("policies.policyTestPanel.simulate")}
            </Button>
            <Button variant="secondary" onClick={handleReset}>
              {t("common.reset")}
            </Button>
          </div>
        </Form>
      </div>

      {!hasSearched && (
        <div className="bg-white border rounded-lg p-8 text-center">
          <div className="text-gray-400 mb-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-10 w-10 mx-auto mb-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
              />
            </svg>
          </div>
          <p className="text-sm font-medium text-gray-600 mb-1">{t("policies.policyTestPanel.noSimulationYet")}</p>
          <p className="text-xs text-gray-400">{t("policies.policyTestPanel.noSimulationHint")}</p>
        </div>
      )}

      {hasSearched && result && (
        <div className="bg-white border rounded-lg p-6">
          {result.matched_policies.length === 0 ? (
            <Empty description={t("policies.policyTestPanel.noPoliciesMatched")} />
          ) : (
            <>
              <div className="mb-4">
                <p className="text-sm font-semibold mb-2">{t("policies.policyTestPanel.effectiveGuardrails")}</p>
                <div className="flex flex-wrap gap-1">
                  {result.effective_guardrails.length > 0 ? (
                    result.effective_guardrails.map((g) => (
                      <Tag key={g} color="green">
                        {g}
                      </Tag>
                    ))
                  ) : (
                    <span className="text-gray-400 text-sm">{t("common.none")}</span>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">{t("policies.policyTestPanel.matchedPolicies")}</p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 pr-4">{t("policies.policyTestPanel.colPolicy")}</th>
                      <th className="text-left py-2 pr-4">{t("policies.policyTestPanel.colMatchedVia")}</th>
                      <th className="text-left py-2">{t("policies.policyTestPanel.colGuardrailsAdded")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.matched_policies.map((p) => (
                      <tr key={p.policy_name} className="border-b last:border-0">
                        <td className="py-2 pr-4 font-medium">{p.policy_name}</td>
                        <td className="py-2 pr-4">
                          <Tag color="blue">{p.matched_via}</Tag>
                        </td>
                        <td className="py-2">
                          {p.guardrails_added.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {p.guardrails_added.map((g) => (
                                <Tag key={g} color="green">
                                  {g}
                                </Tag>
                              ))}
                            </div>
                          ) : (
                            <span className="text-gray-400">{t("common.none")}</span>
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
        <Alert
          message={t("common.error")}
          description={t("policies.policyTestPanel.resolveFailed")}
          type="error"
          showIcon
        />
      )}
    </div>
  );
};

export default PolicyTestPanel;
