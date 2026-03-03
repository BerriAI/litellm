"use client";

import React, { useEffect, useState } from "react";
import {
  Button,
  InputNumber,
  Popconfirm,
  Select,
  Slider,
  Switch,
  Tag,
} from "antd";
import {
  ArrowRightOutlined,
  DeleteOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

export interface RolloutConfig {
  id: string;
  liveModel: string;
  candidateModel: string;
  canaryPercent: number;
  autoRollback: boolean;
  errorThreshold: number;
}

const ROLLOUT_STORAGE_KEY = "litellm_model_rollout_configs";

export function getRolloutConfigs(): RolloutConfig[] {
  try {
    const raw = localStorage.getItem(ROLLOUT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function findRolloutForModel(model: string): RolloutConfig | null {
  const configs = getRolloutConfigs();
  return configs.find(
    (c) =>
      c.liveModel &&
      c.candidateModel &&
      c.liveModel !== c.candidateModel &&
      (c.liveModel === model || c.candidateModel === model),
  ) ?? null;
}

function saveConfigs(configs: RolloutConfig[]) {
  localStorage.setItem(ROLLOUT_STORAGE_KEY, JSON.stringify(configs));
}

function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}

interface ModelRolloutProps {
  accessToken: string | null;
}

const CANARY_PRESETS = [5, 10, 25, 50, 100];

const ModelRollout: React.FC<ModelRolloutProps> = ({ accessToken }) => {
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [configs, setConfigs] = useState<RolloutConfig[]>([]);

  useEffect(() => {
    if (!accessToken) return;
    fetchAvailableModels(accessToken).then(setModels).catch(console.error);
  }, [accessToken]);

  useEffect(() => {
    setConfigs(getRolloutConfigs());
  }, []);

  const modelOptions = Array.from(new Set(models.map((m) => m.model_group))).map((name) => ({
    value: name,
    label: name,
  }));

  const persist = (next: RolloutConfig[]) => {
    setConfigs(next);
    saveConfigs(next);
  };

  const addNew = () => {
    persist([
      ...configs,
      {
        id: newId(),
        liveModel: "",
        candidateModel: "",
        canaryPercent: 10,
        autoRollback: true,
        errorThreshold: 5,
      },
    ]);
  };

  const update = (id: string, patch: Partial<RolloutConfig>) => {
    persist(configs.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  };

  const remove = (id: string) => {
    persist(configs.filter((c) => c.id !== id));
  };

  const statusFor = (c: RolloutConfig) => {
    if (!c.liveModel || !c.candidateModel) return { label: "Not Configured", color: "default" as const };
    if (c.canaryPercent === 100) return { label: "Fully Promoted", color: "green" as const };
    if (c.canaryPercent === 0) return { label: "Not Started", color: "default" as const };
    return { label: `${c.canaryPercent}% canary`, color: "blue" as const };
  };

  return (
    <div className="w-full max-w-screen p-6 overflow-x-hidden box-border">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold">Model Rollout</h1>
          <span className="text-sm text-gray-500">
            Manage A/B tests across your models &middot; {configs.length} active
          </span>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={addNew}>
          New A/B Test
        </Button>
      </div>

      {configs.length === 0 && (
        <div className="bg-white rounded-lg shadow p-12 text-center text-gray-400">
          No rollout tests configured. Click &ldquo;New A/B Test&rdquo; to get started.
        </div>
      )}

      <div className="flex flex-col gap-4">
        {configs.map((config) => {
          const status = statusFor(config);
          const livePercent = 100 - config.canaryPercent;
          return (
            <div key={config.id} className="bg-white rounded-lg shadow">
              {/* Header */}
              <div className="px-6 py-4 border-b flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Tag
                    color="green"
                    style={{ fontSize: 13, padding: "2px 10px", borderRadius: 12, margin: 0 }}
                  >
                    {config.liveModel || "Select live model"}
                  </Tag>
                  <ArrowRightOutlined style={{ color: "#bbb", fontSize: 12 }} />
                  <Tag
                    color="blue"
                    style={{ fontSize: 13, padding: "2px 10px", borderRadius: 12, margin: 0 }}
                  >
                    {config.candidateModel || "Select candidate"}
                  </Tag>
                  <Tag color={status.color}>{status.label}</Tag>
                </div>
                <Popconfirm
                  title="Delete this A/B test?"
                  onConfirm={() => remove(config.id)}
                  okText="Delete"
                  okButtonProps={{ danger: true }}
                >
                  <Button type="text" danger icon={<DeleteOutlined />} size="small" />
                </Popconfirm>
              </div>

              {/* Body */}
              <div className="p-6">
                <div className="grid grid-cols-3 gap-6 mb-4">
                  <div>
                    <label className="block font-medium text-sm mb-2">Live Model</label>
                    <Select
                      value={config.liveModel || undefined}
                      placeholder="Select live model"
                      onChange={(v) => update(config.id, { liveModel: v })}
                      options={modelOptions}
                      style={{ width: "100%" }}
                      showSearch
                    />
                  </div>
                  <div>
                    <label className="block font-medium text-sm mb-2">Candidate Model</label>
                    <Select
                      value={config.candidateModel || undefined}
                      placeholder="Select candidate"
                      onChange={(v) => update(config.id, { candidateModel: v })}
                      options={modelOptions}
                      style={{ width: "100%" }}
                      showSearch
                    />
                  </div>
                  <div>
                    <label className="block font-medium text-sm mb-2">Auto-Rollback</label>
                    <div className="flex items-center gap-3">
                      <Switch
                        checked={config.autoRollback}
                        onChange={(v) => update(config.id, { autoRollback: v })}
                        size="small"
                      />
                      <span className="text-sm text-gray-500">at</span>
                      <InputNumber
                        min={1}
                        max={100}
                        value={config.errorThreshold}
                        onChange={(v) => update(config.id, { errorThreshold: v ?? 5 })}
                        size="small"
                        style={{ width: 60 }}
                      />
                      <span className="text-sm text-gray-500">% errors</span>
                    </div>
                  </div>
                </div>

                {/* Traffic split */}
                <div>
                  <label className="block font-medium text-sm mb-1">
                    Traffic Split
                    <span className="ml-2 font-normal text-gray-400">
                      {livePercent}% live / {config.canaryPercent}% canary
                    </span>
                  </label>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <Slider
                        min={0}
                        max={100}
                        value={config.canaryPercent}
                        onChange={(v) => update(config.id, { canaryPercent: v })}
                        tooltip={{ formatter: (v) => `${v}% canary` }}
                      />
                    </div>
                    <div className="flex gap-1 shrink-0">
                      {CANARY_PRESETS.map((p) => (
                        <Button
                          key={p}
                          size="small"
                          type={config.canaryPercent === p ? "primary" : "default"}
                          onClick={() => update(config.id, { canaryPercent: p })}
                          style={{ minWidth: 40 }}
                        >
                          {p}%
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelRollout;
