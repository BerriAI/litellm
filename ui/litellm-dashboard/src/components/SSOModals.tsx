import React from "react";
import { Modal, Form, Input, Button as Button2, Select } from "antd";
import { Text, TextInput } from "@tremor/react";

interface SSOModalsProps {
  isAddSSOModalVisible: boolean;
  isInstructionsModalVisible: boolean;
  handleAddSSOOk: () => void;
  handleAddSSOCancel: () => void;
  handleShowInstructions: (formValues: Record<string, any>) => void;
  handleInstructionsOk: () => void;
  handleInstructionsCancel: () => void;
  form: any; // Replace with proper Form type if available
}

const ssoProviderLogoMap: Record<string, string> = {
  google: "https://artificialanalysis.ai/img/logos/google_small.svg",
  microsoft: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
};

// Define the SSO provider configuration type
interface SSOProviderConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    label: string;
    name: string;
    placeholder?: string;
  }>;
}

// Define configurations for each SSO provider
const ssoProviderConfigs: Record<string, SSOProviderConfig> = {
  google: {
    envVarMap: {
      google_client_id: 'GOOGLE_CLIENT_ID',
      google_client_secret: 'GOOGLE_CLIENT_SECRET',
    },
    fields: [
      { label: 'GOOGLE CLIENT ID', name: 'google_client_id' },
      { label: 'GOOGLE CLIENT SECRET', name: 'google_client_secret' },
    ],
  },
  microsoft: {
    envVarMap: {
      microsoft_client_id: 'MICROSOFT_CLIENT_ID',
      microsoft_client_secret: 'MICROSOFT_CLIENT_SECRET',
      microsoft_tenant: 'MICROSOFT_TENANT',
    },
    fields: [
      { label: 'MICROSOFT CLIENT ID', name: 'microsoft_client_id' },
      { label: 'MICROSOFT CLIENT SECRET', name: 'microsoft_client_secret' },
      { label: 'MICROSOFT TENANT', name: 'microsoft_tenant' },
    ],
  },
  okta: {
    envVarMap: {
      generic_client_id: 'GENERIC_CLIENT_ID',
      generic_client_secret: 'GENERIC_CLIENT_SECRET',
      generic_authorization_endpoint: 'GENERIC_AUTHORIZATION_ENDPOINT',
      generic_token_endpoint: 'GENERIC_TOKEN_ENDPOINT',
      generic_userinfo_endpoint: 'GENERIC_USERINFO_ENDPOINT',
    },
    fields: [
      { label: 'GENERIC CLIENT ID', name: 'generic_client_id' },
      { label: 'GENERIC CLIENT SECRET', name: 'generic_client_secret' },
      { label: 'AUTHORIZATION ENDPOINT', name: 'generic_authorization_endpoint', placeholder: 'https://your-okta-domain/authorize' },
      { label: 'TOKEN ENDPOINT', name: 'generic_token_endpoint', placeholder: 'https://your-okta-domain/token' },
      { label: 'USERINFO ENDPOINT', name: 'generic_userinfo_endpoint', placeholder: 'https://your-okta-domain/userinfo' },
    ],
  },
  generic: {
    envVarMap: {
      generic_client_id: 'GENERIC_CLIENT_ID',
      generic_client_secret: 'GENERIC_CLIENT_SECRET',
      generic_authorization_endpoint: 'GENERIC_AUTHORIZATION_ENDPOINT',
      generic_token_endpoint: 'GENERIC_TOKEN_ENDPOINT',
      generic_userinfo_endpoint: 'GENERIC_USERINFO_ENDPOINT',
    },
    fields: [
      { label: 'GENERIC CLIENT ID', name: 'generic_client_id' },
      { label: 'GENERIC CLIENT SECRET', name: 'generic_client_secret' },
      { label: 'AUTHORIZATION ENDPOINT', name: 'generic_authorization_endpoint' },
      { label: 'TOKEN ENDPOINT', name: 'generic_token_endpoint' },
      { label: 'USERINFO ENDPOINT', name: 'generic_userinfo_endpoint' },
    ],
  },
};

const SSOModals: React.FC<SSOModalsProps> = ({
  isAddSSOModalVisible,
  isInstructionsModalVisible,
  handleAddSSOOk,
  handleAddSSOCancel,
  handleShowInstructions,
  handleInstructionsOk,
  handleInstructionsCancel,
  form,
}) => {
  // Helper function to render provider fields
  const renderProviderFields = (provider: string) => {
    const config = ssoProviderConfigs[provider];
    if (!config) return null;

    return config.fields.map((field) => (
      <Form.Item
        key={field.name}
        label={field.label}
        name={field.name}
        rules={[{ required: true, message: `Please enter the ${field.label.toLowerCase()}` }]}
      >
        {field.name.includes('client') ? (
          <Input.Password />
        ) : (
          <TextInput placeholder={field.placeholder} />
        )}
      </Form.Item>
    ));
  };

  return (
    <>
      <Modal
        title="Add SSO"
        visible={isAddSSOModalVisible}
        width={800}
        footer={null}
        onOk={handleAddSSOOk}
        onCancel={handleAddSSOCancel}
      >
        <Form
          form={form}
          onFinish={handleShowInstructions}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item
              label="SSO Provider"
              name="sso_provider"
              rules={[{ required: true, message: "Please select an SSO provider" }]}
            >
              <Select>
                {Object.entries(ssoProviderLogoMap).map(([value, logo]) => (
                  <Select.Option key={value} value={value}>
                    <div style={{ display: 'flex', alignItems: 'center', padding: '4px 0' }}>
                      {logo && <img src={logo} alt={value} style={{ height: 24, width: 24, marginRight: 12, objectFit: 'contain' }} />}
                      <span>{value.charAt(0).toUpperCase() + value.slice(1)} SSO</span>
                    </div>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) => prevValues.sso_provider !== currentValues.sso_provider}
            >
              {({ getFieldValue }) => {
                const provider = getFieldValue('sso_provider');
                return provider ? renderProviderFields(provider) : null;
              }}
            </Form.Item>

            <Form.Item
              label="Proxy Admin Email"
              name="user_email"
              rules={[{ required: true, message: "Please enter the email of the proxy admin" }]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item
              label="PROXY BASE URL"
              name="proxy_base_url"
              rules={[{ required: true, message: "Please enter the proxy base url" }]}
            >
              <TextInput />
            </Form.Item>
          </>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Save</Button2>
          </div>
        </Form>
      </Modal>

      <Modal
        title="SSO Setup Instructions"
        visible={isInstructionsModalVisible}
        width={800}
        footer={null}
        onOk={handleInstructionsOk}
        onCancel={handleInstructionsCancel}
      >
        <p>Follow these steps to complete the SSO setup:</p>
        <Text className="mt-2">1. DO NOT Exit this TAB</Text>
        <Text className="mt-2">2. Open a new tab, visit your proxy base url</Text>
        <Text className="mt-2">
          3. Confirm your SSO is configured correctly and you can login on the new
          Tab
        </Text>
        <Text className="mt-2">
          4. If Step 3 is successful, you can close this tab
        </Text>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 onClick={handleInstructionsOk}>Done</Button2>
        </div>
      </Modal>
    </>
  );
};

export { ssoProviderConfigs };  // Export for use in other components
export default SSOModals; 