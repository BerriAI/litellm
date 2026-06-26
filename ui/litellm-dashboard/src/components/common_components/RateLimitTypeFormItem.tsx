import React from "react";
import { Form, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";

const { Option } = Select;

interface RateLimitTypeFormItemProps {
  /** The type of rate limit - either 'tpm' or 'rpm' */
  type: "tpm" | "rpm";
  /** The form field name */
  name: string;
  /** Whether to show detailed descriptions (default: true) */
  showDetailedDescriptions?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Initial value for the field */
  initialValue?: string | null;
  /** Form instance for setting field values */
  form?: any;
  /** Custom onChange handler */
  onChange?: (value: string) => void;
}

export const RateLimitTypeFormItem: React.FC<RateLimitTypeFormItemProps> = ({
  type,
  name,
  showDetailedDescriptions = true,
  className = "",
  initialValue = null,
  form,
  onChange,
}) => {
  const { t } = useTranslation();
  const limitTypeUpper = type.toUpperCase();
  const limitTypeLower = type.toLowerCase();

  const handleChange = (value: string) => {
    if (form) {
      form.setFieldValue(name, value);
    }
    if (onChange) {
      onChange(value);
    }
  };

  const tooltipTitle = t("commonComponents.rateLimitTypeFormItem.tooltip", {
    limitTypeUpper,
  });

  return (
    <Form.Item
      label={
        <span>
          {t("commonComponents.rateLimitTypeFormItem.label", { limitTypeUpper })}{" "}
          <Tooltip title={tooltipTitle}>
            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
          </Tooltip>
        </span>
      }
      name={name}
      initialValue={initialValue}
      className={className}
    >
      <Select
        defaultValue={showDetailedDescriptions ? "default" : undefined}
        placeholder={t("commonComponents.rateLimitTypeFormItem.selectPlaceholder")}
        style={{ width: "100%" }}
        optionLabelProp={showDetailedDescriptions ? "label" : undefined}
        onChange={handleChange}
      >
        {showDetailedDescriptions ? (
          <>
            <Option value="best_effort_throughput" label={t("common.default")}>
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>{t("common.default")}</div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {t("commonComponents.rateLimitTypeFormItem.bestEffortDesc", { limitTypeLower })}
                </div>
              </div>
            </Option>
            <Option
              value="guaranteed_throughput"
              label={t("commonComponents.rateLimitTypeFormItem.guaranteedThroughput")}
            >
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>
                  {t("commonComponents.rateLimitTypeFormItem.guaranteedThroughput")}
                </div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {t("commonComponents.rateLimitTypeFormItem.guaranteedThroughputDesc", { limitTypeLower })}
                </div>
              </div>
            </Option>
            <Option value="dynamic" label={t("commonComponents.rateLimitTypeFormItem.dynamic")}>
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>{t("commonComponents.rateLimitTypeFormItem.dynamic")}</div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {t("commonComponents.rateLimitTypeFormItem.dynamicDesc", { limitTypeUpper })}
                </div>
              </div>
            </Option>
          </>
        ) : (
          <>
            <Option value="best_effort_throughput">
              {t("commonComponents.rateLimitTypeFormItem.bestEffortThroughput")}
            </Option>
            <Option value="guaranteed_throughput">
              {t("commonComponents.rateLimitTypeFormItem.guaranteedThroughput")}
            </Option>
            <Option value="dynamic">{t("commonComponents.rateLimitTypeFormItem.dynamic")}</Option>
          </>
        )}
      </Select>
    </Form.Item>
  );
};

export default RateLimitTypeFormItem;
