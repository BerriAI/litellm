import React from "react";
import { Modal, Form, InputNumber } from "antd";
import { TextInput } from "@tremor/react";
import { Button as Button2 } from "antd";
import { modelUpdateCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
interface EditModelModalProps {
  visible: boolean;
  onCancel: () => void;
  model: any;
  onSubmit: (data: FormData) => void;
}

export const handleEditModelSubmit = async (
  formValues: Record<string, any>,
  accessToken: string | null,
  setEditModalVisible: (visible: boolean) => void,
  setSelectedModel: (model: any) => void,
) => {
  // Call API to update team with teamId and values

  console.log("handleEditSubmit:", formValues);
  if (accessToken == null) {
    return;
  }

  let newLiteLLMParams: Record<string, any> = {};
  let model_info_model_id = null;

  if (formValues.input_cost_per_token) {
    // Convert from per 1M tokens to per token
    formValues.input_cost_per_token = Number(formValues.input_cost_per_token) / 1_000_000;
  }
  if (formValues.output_cost_per_token) {
    // Convert from per 1M tokens to per token
    formValues.output_cost_per_token = Number(formValues.output_cost_per_token) / 1_000_000;
  }

  for (const [key, value] of Object.entries(formValues)) {
    if (key !== "model_id") {
      // Empty string means user wants to null the value
      newLiteLLMParams[key] = value === "" ? null : value;
    } else {
      model_info_model_id = value === "" ? null : value;
    }
  }

  let payload: {
    litellm_params: Record<string, any> | undefined;
    model_info: { id: any } | undefined;
  } = {
    litellm_params: Object.keys(newLiteLLMParams).length > 0 ? newLiteLLMParams : undefined,
    model_info:
      model_info_model_id !== undefined
        ? {
            id: model_info_model_id,
          }
        : undefined,
  };

  console.log("handleEditSubmit payload:", payload);

  try {
    let newModelValue = await modelUpdateCall(accessToken, payload);
    NotificationsManager.success("Model updated successfully, restart server to see updates");

    setEditModalVisible(false);
    setSelectedModel(null);
  } catch (error) {
    console.log(`Error occurred`);
  }
};

const EditModelModal: React.FC<EditModelModalProps> = ({ visible, onCancel, model, onSubmit }) => {
  const [form] = Form.useForm();
  let litellm_params_to_edit: Record<string, any> = {};
  let model_name = "";
  let model_id = "";
  if (model) {
    litellm_params_to_edit = {
      ...model.litellm_params,
      input_cost_per_token: model.litellm_params?.input_cost_per_token
        ? model.litellm_params.input_cost_per_token * 1_000_000
        : undefined,
      output_cost_per_token: model.litellm_params?.output_cost_per_token
        ? model.litellm_params.output_cost_per_token * 1_000_000
        : undefined,
    };
    model_name = model.model_name;
    let model_info = model.model_info;
    if (model_info) {
      model_id = model_info.id;
      console.log(`model_id: ${model_id}`);
      litellm_params_to_edit.model_id = model_id;
    }
  }

  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        const submissionValues = {
          ...values,
          input_cost_per_token: values.input_cost_per_token
            ? Number(values.input_cost_per_token) / 1_000_000
            : undefined,
          output_cost_per_token: values.output_cost_per_token
            ? Number(values.output_cost_per_token) / 1_000_000
            : undefined,
        };
        onSubmit(submissionValues);
        form.resetFields();
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
  };

  return (
    <Modal
      title={"Edit '" + model_name + "' LiteLLM Params"}
      visible={visible}
      width={800}
      footer={null}
      onOk={handleOk}
      onCancel={onCancel}
    >
      <Form
        form={form}
        onFinish={onSubmit}
        initialValues={litellm_params_to_edit}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
      >
        <>
          <Form.Item
            label="Input Cost (per 1M tokens)"
            name="input_cost_per_token"
            tooltip="float (optional) - Input cost per 1 million tokens"
          >
            <TextInput />
          </Form.Item>

          <Form.Item
            label="Output Cost (per 1M tokens)"
            name="output_cost_per_token"
            tooltip="float (optional) - Output cost per 1 million tokens"
          >
            <TextInput />
          </Form.Item>
          <Form.Item className="mt-8" label="api_base" name="api_base">
            <TextInput />
          </Form.Item>
          <Form.Item className="mt-8" label="api_key" name="api_key">
            <TextInput />
          </Form.Item>
          <Form.Item className="mt-8" label="custom_llm_provider" name="custom_llm_provider">
            <TextInput />
          </Form.Item>
          <Form.Item className="mt-8" label="model" name="model">
            <TextInput />
          </Form.Item>
          <Form.Item label="organization" name="organization" tooltip="OpenAI Organization ID">
            <TextInput />
          </Form.Item>

          <Form.Item
            label="tpm"
            name="tpm"
            tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
          >
            <InputNumber min={0} step={1} />
          </Form.Item>

          <Form.Item
            label="rpm"
            name="rpm"
            tooltip="int (optional) - Rate limit for this deployment: in requests per minute (rpm). Find this information on your model/providers website"
          >
            <InputNumber min={0} step={1} />
          </Form.Item>

          <Form.Item label="max_retries" name="max_retries">
            <InputNumber min={0} step={1} />
          </Form.Item>

          <Form.Item
            label="timeout"
            name="timeout"
            tooltip="int (optional) - Timeout in seconds for LLM requests (Defaults to 600 seconds)"
          >
            <InputNumber min={0} step={1} />
          </Form.Item>

          <Form.Item
            label="stream_timeout"
            name="stream_timeout"
            tooltip="int (optional) - Timeout for stream requests (seconds)"
          >
            <InputNumber min={0} step={1} />
          </Form.Item>

          <Form.Item label="model_id" name="model_id" hidden={true}></Form.Item>
        </>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">Save</Button2>
        </div>
      </Form>
    </Modal>
  );
};

export default EditModelModal;
