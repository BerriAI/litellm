import React from "react";
import { Form, Select } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col, Typography, Button as Button2, Upload, UploadProps } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { Providers } from "../provider_info_helpers";
const { Link } = Typography;

interface ProviderSpecificFieldsProps {
  selectedProvider: Providers;
  uploadProps?: UploadProps;
}

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({
  selectedProvider,
  uploadProps
}) => {
  console.log(`Selected provider: ${selectedProvider}`);
  console.log(`type of selectedProvider: ${typeof selectedProvider}`);
  // cast selectedProvider to Providers
  const selectedProviderEnum = Providers[selectedProvider as keyof typeof Providers] as Providers;
  console.log(`selectedProviderEnum: ${selectedProviderEnum}`);
  console.log(`type of selectedProviderEnum: ${typeof selectedProviderEnum}`);
  return (
    <>
      {selectedProviderEnum === Providers.OpenAI && (
        <Form.Item label="Organization ID" name="organization">
          <TextInput placeholder="[OPTIONAL] my-unique-org" />
        </Form.Item>
      )}

      {selectedProviderEnum === Providers.Vertex_AI && (
        <>
          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="Vertex Project"
            name="vertex_project"
          >
            <TextInput placeholder="adroit-cadet-1234.." />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="Vertex Location"
            name="vertex_location"
          >
            <TextInput placeholder="us-east-1" />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="Vertex Credentials"
            name="vertex_credentials"
            className="mb-0"
          >
            <Upload {...uploadProps}>
              <Button2 icon={<UploadOutlined />}>
                Click to Upload
              </Button2>
            </Upload>
          </Form.Item>

          <Row>
            <Col span={10}></Col>
            <Col span={10}>
              <Text className="mb-3 mt-1">
                Give litellm a gcp service account(.json file), so it
                can make the relevant calls
              </Text>
            </Col>
          </Row>
        </>
      )}

      {selectedProviderEnum === Providers.AssemblyAI && (
        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="API Base"
          name="api_base"
        >
          <Select placeholder="Select API Base">
            <Select.Option value="https://api.assemblyai.com">https://api.assemblyai.com</Select.Option>
            <Select.Option value="https://api.eu.assemblyai.com">https://api.eu.assemblyai.com</Select.Option>
          </Select>
        </Form.Item>
      )}

      {(selectedProviderEnum === Providers.Azure ||
        selectedProviderEnum === Providers.OpenAI_Compatible
      ) && (
        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="API Base"
          name="api_base"
        >
          <TextInput placeholder="https://..." />
        </Form.Item>
      )}

      {selectedProviderEnum === Providers.Azure && (
        <>
          <Form.Item
            label="API Version"
            name="api_version"
            tooltip="By default litellm will use the latest version. If you want to use a different version, you can specify it here"
          >
            <TextInput placeholder="2023-07-01-preview" />
          </Form.Item>

          <div>
            <Form.Item
              label="Base Model"
              name="base_model"
              className="mb-0"
            >
              <TextInput placeholder="azure/gpt-3.5-turbo" />
            </Form.Item>
            <Row>
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="mb-2">
                  The actual model your azure deployment uses. Used
                  for accurate cost tracking. Select name from{" "}
                  <Link
                    href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                    target="_blank"
                  >
                    here
                  </Link>
                </Text>
              </Col>
            </Row>
          </div>
        </>
      )}

      {selectedProviderEnum === Providers.Bedrock && (
        <>
          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="AWS Access Key ID"
            name="aws_access_key_id"
            tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
          >
            <TextInput placeholder="" />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="AWS Secret Access Key"
            name="aws_secret_access_key"
            tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
          >
            <TextInput placeholder="" />
          </Form.Item>

          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="AWS Region Name"
            name="aws_region_name"
            tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
          >
            <TextInput placeholder="us-east-1" />
          </Form.Item>
        </>
      )}

      {selectedProviderEnum != Providers.Bedrock &&
        selectedProviderEnum != Providers.Vertex_AI &&
        selectedProviderEnum != Providers.Ollama &&
        (
          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="API Key"
            name="api_key"
            tooltip="LLM API Credentials"
          >
            <TextInput placeholder="sk-" type="password" />
          </Form.Item>
        )}
    </>
  );
};

export default ProviderSpecificFields;