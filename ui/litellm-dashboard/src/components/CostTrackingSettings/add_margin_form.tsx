import React from "react";
import { TextInput, Button } from "@tremor/react";
import { Select as AntdSelect, Form, Tooltip, Radio } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { Providers, provider_map, providerLogoMap } from "../provider_info_helpers";
import { MarginConfig } from "./types";
import { handleImageError } from "./provider_display_helpers";

interface AddMarginFormProps {
  marginConfig: MarginConfig;
  selectedProvider: string | undefined;
  marginType: "percentage" | "fixed";
  percentageValue: string;
  fixedAmountValue: string;
  onProviderChange: (provider: string | undefined) => void;
  onMarginTypeChange: (type: "percentage" | "fixed") => void;
  onPercentageChange: (value: string) => void;
  onFixedAmountChange: (value: string) => void;
  onAddProvider: () => void;
}

const AddMarginForm: React.FC<AddMarginFormProps> = ({
  marginConfig,
  selectedProvider,
  marginType,
  percentageValue,
  fixedAmountValue,
  onProviderChange,
  onMarginTypeChange,
  onPercentageChange,
  onFixedAmountChange,
  onAddProvider,
}) => {
  const { t } = useTranslation();

  return (
    <div className="space-y-6">
      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            {t("costTracking.addMarginForm.providerLabel")}
            <Tooltip title={t("costTracking.addMarginForm.providerTooltip")}>
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: t("costTracking.addMarginForm.providerRequired") }]}
      >
        <AntdSelect
          showSearch
          placeholder={t("costTracking.addMarginForm.providerPlaceholder")}
          value={selectedProvider}
          onChange={onProviderChange}
          style={{ width: "100%" }}
          size="large"
          optionFilterProp="children"
          filterOption={(input, option) =>
            String(option?.label ?? "")
              .toLowerCase()
              .includes(input.toLowerCase())
          }
        >
          <AntdSelect.Option key="global" value="global" label={t("costTracking.addMarginForm.globalOption")}>
            <div className="flex items-center space-x-2">
              <span className="font-medium">{t("costTracking.addMarginForm.globalOption")}</span>
            </div>
          </AntdSelect.Option>
          {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => {
            const providerValue = provider_map[providerEnum as keyof typeof provider_map];
            // Only show providers that don't already have a margin configured
            if (providerValue && marginConfig[providerValue]) {
              return null;
            }
            return (
              <AntdSelect.Option key={providerEnum} value={providerEnum} label={providerDisplayName}>
                <div className="flex items-center space-x-2">
                  <img
                    src={providerLogoMap[providerDisplayName]}
                    alt={`${providerEnum} logo`}
                    className="w-5 h-5"
                    onError={(e) => handleImageError(e, providerDisplayName)}
                  />
                  <span>{providerDisplayName}</span>
                </div>
              </AntdSelect.Option>
            );
          })}
        </AntdSelect>
      </Form.Item>

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            {t("costTracking.addMarginForm.marginTypeLabel")}
            <Tooltip title={t("costTracking.addMarginForm.marginTypeTooltip")}>
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: t("costTracking.addMarginForm.marginTypeRequired") }]}
      >
        <Radio.Group value={marginType} onChange={(e) => onMarginTypeChange(e.target.value)} className="w-full">
          <Radio value="percentage">{t("costTracking.addMarginForm.percentageBased")}</Radio>
          <Radio value="fixed">{t("costTracking.addMarginForm.fixedAmount")}</Radio>
        </Radio.Group>
      </Form.Item>

      {marginType === "percentage" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-gray-700 flex items-center">
              {t("costTracking.addMarginForm.marginPercentageLabel")}
              <Tooltip title={t("costTracking.addMarginForm.marginPercentageTooltip")}>
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </span>
          }
          rules={[
            { required: true, message: t("costTracking.addMarginForm.percentageRequired") },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(new Error(t("costTracking.addMarginForm.percentageRequired")));
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0 || numValue > 1000) {
                  return Promise.reject(new Error(t("costTracking.addMarginForm.percentageRangeError")));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <TextInput
              placeholder="10"
              value={percentageValue}
              onValueChange={onPercentageChange}
              className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 flex-1"
            />
            <span className="text-gray-600">%</span>
          </div>
        </Form.Item>
      )}

      {marginType === "fixed" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-gray-700 flex items-center">
              {t("costTracking.addMarginForm.fixedAmountLabel")}
              <Tooltip title={t("costTracking.addMarginForm.fixedAmountTooltip")}>
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </span>
          }
          rules={[
            { required: true, message: t("costTracking.addMarginForm.fixedAmountRequired") },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(new Error(t("costTracking.addMarginForm.fixedAmountRequired")));
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0) {
                  return Promise.reject(new Error(t("costTracking.addMarginForm.fixedAmountNonNegative")));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <span className="text-gray-600">$</span>
            <TextInput
              placeholder="0.001"
              value={fixedAmountValue}
              onValueChange={onFixedAmountChange}
              className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 flex-1"
            />
          </div>
        </Form.Item>
      )}

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
        <Button
          variant="primary"
          onClick={onAddProvider}
          disabled={
            !selectedProvider ||
            (marginType === "percentage" && !percentageValue) ||
            (marginType === "fixed" && !fixedAmountValue)
          }
        >
          {t("costTracking.addMarginForm.addButton")}
        </Button>
      </div>
    </div>
  );
};

export default AddMarginForm;
