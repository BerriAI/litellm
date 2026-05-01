"use client";

import React from "react";
import { Form, Select, InputNumber, Input, Tooltip } from "antd";
import { PlusOutlined, QuestionCircleOutlined } from "@ant-design/icons";
import { Button } from "antd";

interface LLMJudgeFieldsProps {
  availableModels: string[];
  form: any;
}

const LLMJudgeFields: React.FC<LLMJudgeFieldsProps> = ({ availableModels, form }) => {
  return (
    <>
      <div
        style={{
          background: "#f6ffed",
          border: "1px solid #b7eb8f",
          borderRadius: 6,
          padding: "10px 14px",
          marginBottom: 16,
          fontSize: 13,
          color: "#389e0d",
        }}
      >
        After each LLM response, the <strong>Judge Model</strong> scores it 0–100 against your criteria.
        If the weighted average falls below the threshold, the response is blocked (or logged).
      </div>

      <Form.Item
        name="judge_model"
        label={
          <span>
            Judge Model&nbsp;
            <Tooltip title="The LLM that reads each response and grades it. Pick a capable model — it never sees end-user data beyond what the LLM returned.">
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: "Select a judge model" }]}
      >
        <Select
          showSearch
          placeholder="Select a model"
          options={availableModels.map((m) => ({ label: m, value: m }))}
        />
      </Form.Item>

      <Form.Item
        name="overall_threshold"
        label={
          <span>
            Minimum Score to Pass&nbsp;
            <Tooltip title="0–100. If the weighted average of criterion scores falls below this, the guardrail triggers. 80 is a good default.">
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
        initialValue={80}
      >
        <InputNumber min={0} max={100} addonAfter="/ 100" style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item
        name="on_failure"
        label={
          <span>
            On Failure&nbsp;
            <Tooltip title="Block: return HTTP 422 when the score is too low. Log: record the result but let the response through.">
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
        initialValue="block"
      >
        <Select>
          <Select.Option value="block">Block (return 422)</Select.Option>
          <Select.Option value="log">Log only</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <span>
            Evaluation Criteria&nbsp;
            <Tooltip title="Each criterion is something the judge checks. Weights must add up to 100%.">
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
      >
        <Form.List name="criteria" initialValue={[{ name: "", weight: 100, description: "" }]}>
          {(fields, { add, remove }) => (
            <>
              {fields.map(({ key, name, ...restField }) => (
                <div
                  key={key}
                  style={{
                    border: "1px solid #f0f0f0",
                    borderRadius: 6,
                    padding: "12px 12px 0",
                    marginBottom: 8,
                  }}
                >
                  <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                    <Form.Item
                      {...restField}
                      name={[name, "name"]}
                      rules={[{ required: true, message: "Enter criterion name" }]}
                      style={{ flex: 2, marginBottom: 8 }}
                    >
                      <Input placeholder="Criterion name (e.g. Policy accuracy)" />
                    </Form.Item>
                    <Form.Item
                      {...restField}
                      name={[name, "weight"]}
                      label={
                        <Tooltip title="How much this criterion counts toward the final score. All weights must add up to 100%.">
                          <span style={{ fontSize: 12, color: "#595959" }}>
                            Weight <QuestionCircleOutlined style={{ color: "#bfbfbf" }} />
                          </span>
                        </Tooltip>
                      }
                      rules={[{ required: true, message: "Enter weight" }]}
                      style={{ flex: 1, marginBottom: 8 }}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        addonAfter="%"
                        style={{ width: "100%" }}
                        placeholder="e.g. 50"
                      />
                    </Form.Item>
                    <div style={{ marginBottom: 8 }}>
                      <Button
                        type="text"
                        danger
                        size="small"
                        onClick={() => remove(name)}
                      >
                        ×
                      </Button>
                    </div>
                  </div>
                  <Form.Item
                    {...restField}
                    name={[name, "description"]}
                    rules={[{ required: true, message: "Describe what to check" }]}
                    style={{ marginBottom: 8 }}
                  >
                    <Input placeholder="What should the judge check for this criterion?" />
                  </Form.Item>
                </div>
              ))}
              <Button
                type="dashed"
                block
                style={{ marginTop: 4 }}
                onClick={() => add({ name: "", weight: 0, description: "" })}
                icon={<PlusOutlined />}
              >
                Add Criterion
              </Button>
              {fields.length > 0 && (
                <Form.Item shouldUpdate noStyle>
                  {() => {
                    const allCriteria: any[] = form.getFieldValue("criteria") || [];
                    const weightTotal = allCriteria.reduce(
                      (sum: number, c: any) => sum + (Number(c?.weight) || 0),
                      0,
                    );
                    const weightOk = weightTotal === 100;
                    return (
                      <div style={{ marginTop: 6, fontSize: 12, color: weightOk ? "#52c41a" : "#faad14" }}>
                        Weights total: {weightTotal}%{weightOk ? " ✓" : " — must add up to 100%"}
                      </div>
                    );
                  }}
                </Form.Item>
              )}
            </>
          )}
        </Form.List>
      </Form.Item>
    </>
  );
};

export default LLMJudgeFields;
