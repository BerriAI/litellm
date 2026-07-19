"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button, Form, Input, Select } from "antd";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { modelCreateCall, modelInfoCall } from "@/components/networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  autoRoutersOf,
  buildComplexityAutorouterPayload,
  COMPLEXITY_TIERS,
  ComplexityTier,
  ComplexityTiers,
  DeploymentListItem,
  DeploymentListResponse,
  emptyComplexityTiers,
} from "./helpers";

interface AutorouterTabProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

const TIER_LABELS: Record<ComplexityTier, string> = {
  SIMPLE: "Simple",
  MEDIUM: "Medium",
  COMPLEX: "Complex",
  REASONING: "Reasoning",
};

const AutorouterTab: React.FC<AutorouterTabProps> = ({ accessToken, userId, userRole }) => {
  const [autoRouters, setAutoRouters] = useState<DeploymentListItem[]>([]);
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [name, setName] = useState<string>("");
  const [defaultModel, setDefaultModel] = useState<string | undefined>(undefined);
  const [tiers, setTiers] = useState<ComplexityTiers>(emptyComplexityTiers());
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadAutoRouters = useCallback(() => {
    if (!accessToken || !userId || !userRole) {
      return;
    }
    modelInfoCall(accessToken, userId, userRole, 1, 100)
      .then((response) => setAutoRouters(autoRoutersOf(response as DeploymentListResponse)))
      .catch((error) => {
        console.error("Failed to load auto routers:", error);
        NotificationsManager.fromBackend("Failed to load auto routers");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken, userId, userRole]);

  useEffect(() => {
    loadAutoRouters();
  }, [loadAutoRouters]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    fetchAvailableModels(accessToken)
      .then((available) => setModels(available.filter((model) => model.mode !== "embedding")))
      .catch((error) => console.error("Error fetching models:", error));
  }, [accessToken]);

  const modelOptions = models.map((model) => ({ value: model.model_group, label: model.model_group }));

  const handleTierChange = (tier: ComplexityTier, selected: string[]) => {
    setTiers((prev) => ({ ...prev, [tier]: selected }));
  };

  const resetForm = () => {
    setName("");
    setDefaultModel(undefined);
    setTiers(emptyComplexityTiers());
  };

  const handleAdd = async () => {
    if (!accessToken) {
      return;
    }
    if (name.trim() === "" || !defaultModel) {
      NotificationsManager.fromBackend("Name and default model are required");
      return;
    }
    setIsSaving(true);
    try {
      await modelCreateCall(accessToken, buildComplexityAutorouterPayload({ name, defaultModel, tiers }));
      NotificationsManager.success("Auto router created");
      resetForm();
      loadAutoRouters();
    } catch (error) {
      console.error("Failed to create auto router:", error);
      NotificationsManager.fromBackend("Failed to create auto router");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Auto routers</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
          {!isLoading && autoRouters.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No auto routers configured yet. Add one below to route each request to the cheapest capable model
            </p>
          )}
          {!isLoading && autoRouters.length > 0 && (
            <ul className="divide-y divide-gray-200">
              {autoRouters.map((router) => (
                <li key={router.model_info?.id ?? router.model_name} className="flex items-center justify-between py-3">
                  <p className="text-sm font-medium text-foreground">{router.model_name}</p>
                  <p className="text-xs text-muted-foreground">
                    {router.litellm_params?.complexity_router_default_model ?? ""}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Add complexity auto router</CardTitle>
        </CardHeader>
        <CardContent>
          <Form layout="vertical" requiredMark={false}>
            <Form.Item label="Name" required>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="cost-router" />
            </Form.Item>
            <Form.Item label="Default model" required>
              <Select
                value={defaultModel}
                onChange={setDefaultModel}
                options={modelOptions}
                showSearch
                placeholder="Fallback model when no tier matches"
              />
            </Form.Item>
            {COMPLEXITY_TIERS.map((tier) => (
              <Form.Item key={tier} label={`${TIER_LABELS[tier]} tier`}>
                <Select
                  mode="multiple"
                  value={tiers[tier]}
                  onChange={(selected: string[]) => handleTierChange(tier, selected)}
                  options={modelOptions}
                  showSearch
                  placeholder={`Model(s) for ${TIER_LABELS[tier].toLowerCase()} requests`}
                />
              </Form.Item>
            ))}
            <div className="flex justify-end">
              <Button type="primary" onClick={handleAdd} loading={isSaving}>
                Add auto router
              </Button>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
};

export default AutorouterTab;
