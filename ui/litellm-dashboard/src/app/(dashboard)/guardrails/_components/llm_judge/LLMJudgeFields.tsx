"use client";

import React from "react";
import { useTranslation } from "react-i18next";
import { Form, Select, InputNumber, Input, Tooltip } from "antd";
import { PlusOutlined, QuestionCircleOutlined } from "@ant-design/icons";
import { Button } from "antd";

interface LLMJudgeFieldsProps {
  availableModels: string[];
  form: any;
}

const LLMJudgeFields: React.FC<LLMJudgeFieldsProps> = ({ availableModels, form }) => {
  const { t } = useTranslation("gateway");
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
        {t("guardrailsPage.llmJudge.description")}
      </div>

      <Form.Item
        name="judge_model"
        label={
          <span>
            {t("guardrailsPage.llmJudge.judgeModel")}&nbsp;
            <Tooltip title={t("guardrailsPage.llmJudge.judgeModelHelp")}>
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: t("guardrailsPage.llmJudge.selectJudgeModel") }]}
      >
        <Select
          showSearch
          placeholder={t("guardrailsPage.llmJudge.selectModel")}
          options={availableModels.map((m) => ({ label: m, value: m }))}
        />
      </Form.Item>

      <Form.Item
        name="overall_threshold"
        label={
          <span>
            {t("guardrailsPage.llmJudge.minimumScore")}&nbsp;
            <Tooltip title={t("guardrailsPage.llmJudge.minimumScoreHelp")}>
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
            {t("guardrailsPage.llmJudge.onFailure")}&nbsp;
            <Tooltip title={t("guardrailsPage.llmJudge.onFailureHelp")}>
              <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
            </Tooltip>
          </span>
        }
        initialValue="block"
      >
        <Select>
          <Select.Option value="block">{t("guardrailsPage.llmJudge.block")}</Select.Option>
          <Select.Option value="log">{t("guardrailsPage.llmJudge.logOnly")}</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <span>
            {t("guardrailsPage.llmJudge.criteria")}&nbsp;
            <Tooltip title={t("guardrailsPage.llmJudge.criteriaHelp")}>
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
                      rules={[{ required: true, message: t("guardrailsPage.llmJudge.criterionNameRequired") }]}
                      style={{ flex: 2, marginBottom: 8 }}
                    >
                      <Input placeholder={t("guardrailsPage.llmJudge.criterionNamePlaceholder")} />
                    </Form.Item>
                    <Form.Item
                      {...restField}
                      name={[name, "weight"]}
                      label={
                        <Tooltip title={t("guardrailsPage.llmJudge.weightHelp")}>
                          <span style={{ fontSize: 12, color: "#595959" }}>
                            {t("guardrailsPage.llmJudge.weight")}{" "}
                            <QuestionCircleOutlined style={{ color: "#bfbfbf" }} />
                          </span>
                        </Tooltip>
                      }
                      rules={[{ required: true, message: t("guardrailsPage.llmJudge.weightRequired") }]}
                      style={{ flex: 1, marginBottom: 8 }}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        addonAfter="%"
                        style={{ width: "100%" }}
                        placeholder={t("guardrailsPage.llmJudge.weightPlaceholder")}
                      />
                    </Form.Item>
                    <div style={{ marginBottom: 8 }}>
                      <Button type="text" danger size="small" onClick={() => remove(name)}>
                        ×
                      </Button>
                    </div>
                  </div>
                  <Form.Item
                    {...restField}
                    name={[name, "description"]}
                    rules={[{ required: true, message: t("guardrailsPage.llmJudge.descriptionRequired") }]}
                    style={{ marginBottom: 8 }}
                  >
                    <Input placeholder={t("guardrailsPage.llmJudge.descriptionPlaceholder")} />
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
                {t("guardrailsPage.llmJudge.addCriterion")}
              </Button>
              {fields.length > 0 && (
                <Form.Item shouldUpdate noStyle>
                  {() => {
                    const allCriteria: any[] = form.getFieldValue("criteria") || [];
                    const weightTotal = allCriteria.reduce((sum: number, c: any) => sum + (Number(c?.weight) || 0), 0);
                    const weightOk = weightTotal === 100;
                    return (
                      <div style={{ marginTop: 6, fontSize: 12, color: weightOk ? "#52c41a" : "#faad14" }}>
                        {t("guardrailsPage.llmJudge.weightsTotal", { weight: weightTotal })}
                        {weightOk ? " ✓" : t("guardrailsPage.llmJudge.weightsInvalid")}
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
