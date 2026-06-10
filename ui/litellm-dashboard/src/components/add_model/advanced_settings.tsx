import React from "react";
import { Form, Switch, Select, Tooltip } from "antd";
import { Text, Accordion, AccordionHeader, AccordionBody, TextInput } from "@tremor/react";
import { Row, Col, Typography } from "antd";
import TextArea from "antd/es/input/TextArea";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useTranslation, Trans } from "react-i18next";
import { Team } from "../key_team_helpers/key_list";
import CacheControlSettings from "./cache_control_settings";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import { Tag } from "../tag_management/types";
import { formItemValidateJSON } from "../../utils/textUtils";
const { Link } = Typography;

interface AdvancedSettingsProps {
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams?: Team[] | null;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  accessToken: string;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  guardrailsList,
  tagsList,
  accessToken,
}) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [customPricing, setCustomPricing] = React.useState(false);
  const [pricingModel, setPricingModel] = React.useState<"per_token" | "per_second">("per_token");
  const [showCacheControl, setShowCacheControl] = React.useState(false);

  const validateNumber = (_: any, value: string) => {
    if (!value) {
      return Promise.resolve();
    }
    if (isNaN(Number(value)) || Number(value) < 0) {
      return Promise.reject(t("addModel.advancedSettings.validateNumberError"));
    }
    return Promise.resolve();
  };

  // Handle custom pricing changes
  const handleCustomPricingChange = (checked: boolean) => {
    setCustomPricing(checked);
    if (!checked) {
      // Clear pricing fields when disabled
      form.setFieldsValue({
        input_cost_per_token: undefined,
        output_cost_per_token: undefined,
        cache_read_input_token_cost: undefined,
        cache_creation_input_token_cost: undefined,
        input_cost_per_second: undefined,
      });
    }
  };

  const handlePassThroughChange = (checked: boolean) => {
    const currentParams = form.getFieldValue("litellm_extra_params");
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (checked) {
        paramsObj.use_in_pass_through = true;
      } else {
        delete paramsObj.use_in_pass_through;
      }
      // Only set the field value if there are remaining parameters
      if (Object.keys(paramsObj).length > 0) {
        form.setFieldValue("litellm_extra_params", JSON.stringify(paramsObj, null, 2));
      } else {
        form.setFieldValue("litellm_extra_params", "");
      }
    } catch (error) {
      // If JSON parsing fails, only create new object if checked is true
      if (checked) {
        form.setFieldValue("litellm_extra_params", JSON.stringify({ use_in_pass_through: true }, null, 2));
      } else {
        form.setFieldValue("litellm_extra_params", "");
      }
    }
  };

  const handleCacheControlChange = (checked: boolean) => {
    setShowCacheControl(checked);
    if (!checked) {
      const currentParams = form.getFieldValue("litellm_extra_params");
      try {
        let paramsObj = currentParams ? JSON.parse(currentParams) : {};
        delete paramsObj.cache_control_injection_points;
        if (Object.keys(paramsObj).length > 0) {
          form.setFieldValue("litellm_extra_params", JSON.stringify(paramsObj, null, 2));
        } else {
          form.setFieldValue("litellm_extra_params", "");
        }
      } catch (error) {
        form.setFieldValue("litellm_extra_params", "");
      }
    }
  };

  return (
    <>
      <Accordion className="mt-2 mb-4">
        <AccordionHeader>
          <b>{t("addModel.advancedSettings.title")}</b>
        </AccordionHeader>
        <AccordionBody>
          <div className="bg-white rounded-lg">
            <Form.Item
              label={t("addModel.advancedSettings.customPricingLabel")}
              name="custom_pricing"
              valuePropName="checked"
              className="mb-4"
            >
              <Switch onChange={handleCustomPricingChange} className="bg-gray-600" />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  {t("addModel.advancedSettings.attachedKnowledgeBasesLabel")}{" "}
                  <Tooltip title={t("addModel.advancedSettings.attachedKnowledgeBasesTooltip")}>
                    <a
                      href="https://docs.litellm.ai/docs/completion/knowledgebase"
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </a>
                  </Tooltip>
                </span>
              }
              name="vector_store_ids"
              className="mt-4"
              help={t("addModel.advancedSettings.attachedKnowledgeBasesHelp")}
            >
              <VectorStoreSelector
                onChange={() => {}}
                accessToken={accessToken}
                placeholder={t("addModel.advancedSettings.knowledgeBasesPlaceholder")}
              />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  {t("addModel.advancedSettings.guardrailsLabel")}{" "}
                  <Tooltip title={t("addModel.advancedSettings.guardrailsTooltip")}>
                    <a
                      href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </a>
                  </Tooltip>
                </span>
              }
              name="guardrails"
              className="mt-4"
              help={t("addModel.advancedSettings.guardrailsHelp")}
            >
              <Select
                mode="tags"
                style={{ width: "100%" }}
                placeholder={t("addModel.advancedSettings.guardrailsPlaceholder")}
                options={guardrailsList.map((name) => ({ value: name, label: name }))}
              />
            </Form.Item>

            <Form.Item label={t("addModel.advancedSettings.tagsLabel")} name="tags" className="mb-4">
              <Select
                mode="tags"
                style={{ width: "100%" }}
                placeholder={t("addModel.advancedSettings.tagsPlaceholder")}
                options={Object.values(tagsList).map((tag) => ({
                  value: tag.name,
                  label: tag.name,
                  title: tag.description || tag.name,
                }))}
              />
            </Form.Item>

            {customPricing && (
              <div className="ml-6 pl-4 border-l-2 border-gray-200">
                <Form.Item
                  label={t("addModel.advancedSettings.pricingModelLabel")}
                  name="pricing_model"
                  className="mb-4"
                >
                  <Select
                    defaultValue="per_token"
                    onChange={(value: "per_token" | "per_second") => setPricingModel(value)}
                    options={[
                      { value: "per_token", label: t("addModel.advancedSettings.perMillionTokens") },
                      { value: "per_second", label: t("addModel.advancedSettings.perSecond") },
                    ]}
                  />
                </Form.Item>

                {pricingModel === "per_token" ? (
                  <>
                    <Form.Item
                      label={t("addModel.advancedSettings.inputCostLabel")}
                      name="input_cost_per_token"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput />
                    </Form.Item>
                    <Form.Item
                      label={t("addModel.advancedSettings.outputCostLabel")}
                      name="output_cost_per_token"
                      rules={[{ validator: validateNumber }]}
                      className="mb-4"
                    >
                      <TextInput />
                    </Form.Item>
                    <Form.Item
                      label={t("addModel.advancedSettings.cacheReadCostLabel")}
                      name="cache_read_input_token_cost"
                      rules={[{ validator: validateNumber }]}
                      tooltip={t("addModel.advancedSettings.cacheReadCostTooltip")}
                      className="mb-4"
                    >
                      <TextInput placeholder={t("addModel.advancedSettings.cacheReadCostPlaceholder")} />
                    </Form.Item>
                    <Form.Item
                      label={t("addModel.advancedSettings.cacheWriteCostLabel")}
                      name="cache_creation_input_token_cost"
                      rules={[{ validator: validateNumber }]}
                      tooltip={t("addModel.advancedSettings.cacheWriteCostTooltip")}
                      className="mb-4"
                    >
                      <TextInput placeholder={t("addModel.advancedSettings.cacheWriteCostPlaceholder")} />
                    </Form.Item>
                  </>
                ) : (
                  <Form.Item
                    label={t("addModel.advancedSettings.costPerSecondLabel")}
                    name="input_cost_per_second"
                    rules={[{ validator: validateNumber }]}
                    className="mb-4"
                  >
                    <TextInput />
                  </Form.Item>
                )}
              </div>
            )}

            <Form.Item
              label={t("addModel.advancedSettings.useInPassThroughLabel")}
              name="use_in_pass_through"
              valuePropName="checked"
              className="mb-4 mt-4"
              tooltip={
                <Trans
                  i18nKey="addModel.advancedSettings.useInPassThroughTooltip"
                  components={{
                    learnMoreLink: <Link href="https://docs.litellm.ai/docs/pass_through/vertex_ai" target="_blank" />,
                  }}
                />
              }
            >
              <Switch onChange={handlePassThroughChange} className="bg-gray-600" />
            </Form.Item>

            <CacheControlSettings
              form={form}
              showCacheControl={showCacheControl}
              onCacheControlChange={handleCacheControlChange}
            />
            <Form.Item
              label={t("addModel.advancedSettings.litellmParamsLabel")}
              name="litellm_extra_params"
              tooltip={t("addModel.advancedSettings.litellmParamsTooltip")}
              className="mb-4 mt-4"
              rules={[{ validator: formItemValidateJSON }]}
            >
              <TextArea
                rows={4}
                placeholder='{
                  "rpm": 100,
                  "timeout": 0,
                  "stream_timeout": 0
                }'
              />
            </Form.Item>
            <Row className="mb-4">
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="text-gray-600 text-sm">
                  {t("addModel.advancedSettings.litellmParamsHelpText")}{" "}
                  <Link href="https://docs.litellm.ai/docs/completion/input" target="_blank">
                    {t("addModel.advancedSettings.litellmParamsHelpLink")}
                  </Link>
                </Text>
              </Col>
            </Row>
            <Form.Item
              label={t("addModel.advancedSettings.modelInfoLabel")}
              name="model_info_params"
              tooltip={t("addModel.advancedSettings.modelInfoTooltip")}
              className="mb-0"
              rules={[{ validator: formItemValidateJSON }]}
            >
              <TextArea
                rows={4}
                placeholder='{
                  "mode": "chat"
                }'
              />
            </Form.Item>
          </div>
        </AccordionBody>
      </Accordion>
    </>
  );
};

export default AdvancedSettings;
