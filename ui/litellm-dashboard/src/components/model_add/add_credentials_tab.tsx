import React, { useState } from "react";
import { 
  Card, 
  Form, 
  Button, 
  Tooltip, 
  Typography, 
  Select as AntdSelect, 
  Input, 
  Switch, 
  Modal 
} from "antd";
import type { UploadProps } from "antd/es/upload";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import type { FormInstance } from "antd";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { TextInput } from "@tremor/react";
const { Title, Link } = Typography;

interface AddCredentialsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({
  isVisible,
  onCancel,
  onAddCredential,
  uploadProps
}) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const handleSubmit = (values: any) => {
    onAddCredential(values);
    form.resetFields();
  };

  return (
    <Modal
      title="Add New Credential"
      visible={isVisible}
      onCancel={() => {
        onCancel();
        form.resetFields();
      }}
      footer={null}
      width={600}
    >
      <Form
        form={form}
        onFinish={handleSubmit}
        layout="vertical"
      >
        {/* Provider Selection */}
        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="Provider:"
          name="custom_llm_provider"
          tooltip="Select the credential provider"
        >
          <AntdSelect
            showSearch={true}
            value={selectedProvider}
            onChange={(value) => {
              setSelectedProvider(value);
              form.setFieldsValue({ 
                model: [],
                model_name: undefined 
              });
            }}
          >
            {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
              <AntdSelect.Option
                key={providerEnum}
                value={providerEnum}
              >
                <div className="flex items-center space-x-2">
                  <img
                    src={providerLogoMap[providerDisplayName]}
                    alt={`${providerEnum} logo`}
                    className="w-5 h-5"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      const parent = target.parentElement;
                      if (parent) {
                        const fallbackDiv = document.createElement('div');
                        fallbackDiv.className = 'w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                        fallbackDiv.textContent = providerDisplayName.charAt(0);
                        parent.replaceChild(fallbackDiv, target);
                      }
                    }}
                  />
                  <span>{providerDisplayName}</span>
                </div>
              </AntdSelect.Option>
            ))}
          </AntdSelect>
        </Form.Item>

        {/* Credential Name */}
        <Form.Item
          label="Credential Name:"
          name="credential_name"
          rules={[{ required: true, message: "Credential name is required" }]}
        >
          <TextInput placeholder="Enter a friendly name for these credentials" />
        </Form.Item>

        <ProviderSpecificFields
          selectedProvider={selectedProvider}
          uploadProps={uploadProps}
        />

        {/* Modal Footer */}
        <div className="flex justify-between items-center">
          <Tooltip title="Get help on our github">
            <Link href="https://github.com/BerriAI/litellm/issues">
              Need Help?
            </Link>
          </Tooltip>
          
          <div>
            <Button 
              onClick={() => {
                onCancel();
                form.resetFields();
              }} 
              style={{ marginRight: 10 }}
            >
              Cancel
            </Button>
            <Button 
              type="primary" 
              htmlType="submit"
            >
              Add Credential
            </Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;