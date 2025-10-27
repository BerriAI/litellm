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
import { CredentialItem } from "../networking";
const { Title, Link } = Typography;

interface ReuseCredentialsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  existingCredential: CredentialItem | null;
  setIsCredentialModalOpen: (isVisible: boolean) => void;
}

const ReuseCredentialsModal: React.FC<ReuseCredentialsModalProps> = ({
  isVisible,
  onCancel,
  onAddCredential,
  existingCredential,
  setIsCredentialModalOpen
}) => {
  const [form] = Form.useForm();

  console.log(`existingCredential in add credentials tab: ${JSON.stringify(existingCredential)}`);

  const handleSubmit = (values: any) => {
    onAddCredential(values);
    form.resetFields();
    setIsCredentialModalOpen(false);
  };

  return (
    <Modal
      title="Reuse Credentials"
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
        {/* Credential Name */}
        <Form.Item
          label="Credential Name:"
          name="credential_name"
          rules={[{ required: true, message: "Credential name is required" }]}
          initialValue={existingCredential?.credential_name}
        >
          <TextInput 
            placeholder="Enter a friendly name for these credentials"
          />
        </Form.Item>

        {/* Display Credential Values of existingCredential, don't allow user to edit. Credential values is a dictionary */}
        {Object.entries(existingCredential?.credential_values || {}).map(([key, value]) => (
          <Form.Item
            key={key}
            label={key}
            name={key}
            initialValue={value}    
          >
            <TextInput 
              placeholder={`Enter ${key}`}
              disabled={true}
            />
          </Form.Item>
        ))}

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
              htmlType="submit"
            >
              Reuse Credentials
            </Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default ReuseCredentialsModal;