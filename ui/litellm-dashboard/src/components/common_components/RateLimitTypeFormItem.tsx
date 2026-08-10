import React from "react";
import { Form, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

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
  labels?: {
    fieldLabel: string;
    tooltip: string;
    placeholder: string;
    defaultLabel: string;
    defaultDescription: string;
    guaranteedLabel: string;
    guaranteedDescription: string;
    dynamicLabel: string;
    dynamicDescription: string;
    bestEffortLabel: string;
  };
}

export const RateLimitTypeFormItem: React.FC<RateLimitTypeFormItemProps> = ({
  type,
  name,
  showDetailedDescriptions = true,
  className = "",
  initialValue = null,
  form,
  onChange,
  labels,
}) => {
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

  const tooltipTitle =
    labels?.tooltip ??
    `Select 'guaranteed_throughput' to prevent overallocating ${limitTypeUpper} limit when the key belongs to a Team with specific ${limitTypeUpper} limits.`;

  return (
    <Form.Item
      label={
        <span>
          {labels?.fieldLabel ?? `${limitTypeUpper} Rate Limit Type`}{" "}
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
        placeholder={labels?.placeholder ?? "Select rate limit type"}
        style={{ width: "100%" }}
        optionLabelProp={showDetailedDescriptions ? "label" : undefined}
        onChange={handleChange}
      >
        {showDetailedDescriptions ? (
          <>
            <Option value="best_effort_throughput" label={labels?.defaultLabel ?? "Default"}>
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>{labels?.defaultLabel ?? "Default"}</div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {labels?.defaultDescription ?? (
                    <>
                      Best effort throughput - no error if we&apos;re overallocating {limitTypeLower} (Team/Key Limits
                      checked at runtime).
                    </>
                  )}
                </div>
              </div>
            </Option>
            <Option value="guaranteed_throughput" label={labels?.guaranteedLabel ?? "Guaranteed throughput"}>
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>{labels?.guaranteedLabel ?? "Guaranteed throughput"}</div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {labels?.guaranteedDescription ?? (
                    <>
                      Guaranteed throughput - raise an error if we&apos;re overallocating {limitTypeLower} (also checks
                      model-specific limits)
                    </>
                  )}
                </div>
              </div>
            </Option>
            <Option value="dynamic" label={labels?.dynamicLabel ?? "Dynamic"}>
              <div style={{ padding: "4px 0" }}>
                <div style={{ fontWeight: 500 }}>{labels?.dynamicLabel ?? "Dynamic"}</div>
                <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                  {labels?.dynamicDescription ?? (
                    <>
                      If the key has a set {limitTypeUpper} (e.g. 2 {limitTypeUpper}) and there are no 429 errors, it
                      can dynamically exceed the limit when the model being called is not erroring.
                    </>
                  )}
                </div>
              </div>
            </Option>
          </>
        ) : (
          <>
            <Option value="best_effort_throughput">{labels?.bestEffortLabel ?? "Best effort throughput"}</Option>
            <Option value="guaranteed_throughput">{labels?.guaranteedLabel ?? "Guaranteed throughput"}</Option>
            <Option value="dynamic">{labels?.dynamicLabel ?? "Dynamic"}</Option>
          </>
        )}
      </Select>
    </Form.Item>
  );
};

export default RateLimitTypeFormItem;
