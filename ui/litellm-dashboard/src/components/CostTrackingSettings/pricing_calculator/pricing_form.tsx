import React from "react";
import { Form, InputNumber, Select, Row, Col } from "antd";
import { PricingFormValues } from "./types";

interface PricingFormProps {
  models: string[];
  onValuesChange: (changedValues: Partial<PricingFormValues>, allValues: PricingFormValues) => void;
}

const PricingForm: React.FC<PricingFormProps> = ({ models, onValuesChange }) => {
  return (
    <Form
      layout="vertical"
      onValuesChange={onValuesChange}
      initialValues={{
        input_tokens: 1000,
        output_tokens: 500,
      }}
    >
      <Row gutter={16}>
        <Col span={12}>
          <Form.Item
            name="model"
            label="Model"
            rules={[{ required: true, message: "Please select a model" }]}
          >
            <Select
              showSearch
              placeholder="Select a model"
              optionFilterProp="label"
              filterOption={(input, option) =>
                String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
              options={models.map((model) => ({
                value: model,
                label: model,
              }))}
            />
          </Form.Item>
        </Col>
        <Col span={6}>
          <Form.Item
            name="input_tokens"
            label="Input Tokens (per request)"
            rules={[{ required: true, message: "Required" }]}
          >
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            />
          </Form.Item>
        </Col>
        <Col span={6}>
          <Form.Item
            name="output_tokens"
            label="Output Tokens (per request)"
            rules={[{ required: true, message: "Required" }]}
          >
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            />
          </Form.Item>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          <Form.Item
            name="num_requests_per_day"
            label="Requests per Day"
            tooltip="Optional: Enter expected daily request volume"
          >
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              placeholder="e.g., 1000"
              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            name="num_requests_per_month"
            label="Requests per Month"
            tooltip="Optional: Enter expected monthly request volume"
          >
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              placeholder="e.g., 30000"
              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
            />
          </Form.Item>
        </Col>
      </Row>
    </Form>
  );
};

export default PricingForm;

