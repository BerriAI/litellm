import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button as Button2, Select, message, Spin, Card, Typography, Popconfirm } from "antd";
import { Text, TextInput } from "@tremor/react";
import { 
  getSSOProviderConfig, 
  updateSSOProviderConfig, 
  deleteSSOProviderConfig,
  type SSOConfigRequest,
  type SSOProviderConfig 
} from "./networking";
import { DeleteOutlined, EditOutlined, EyeOutlined } from "@ant-design/icons";

const { Title, Paragraph } = Typography;

interface SSOModalsProps {
  isAddSSOModalVisible: boolean;
  isInstructionsModalVisible: boolean;
  handleAddSSOOk: () => void;
  handleAddSSOCancel: () => void;
  handleShowInstructions: (formValues: Record<string, any>) => void;
  handleInstructionsOk: () => void;
  handleInstructionsCancel: () => void;
  form: any; // Replace with proper Form type if available
  accessToken: string;
}

const ssoProviderLogoMap: Record<string, string> = {
  google: "https://artificialanalysis.ai/img/logos/google_small.svg",
  microsoft: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
};

// Define the SSO provider field configuration type  
interface SSOProviderFieldConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    label: string;
    name: string;
    placeholder?: string;
  }>;
}

// Define configurations for each SSO provider
const ssoProviderConfigs: Record<string, SSOProviderFieldConfig> = {
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
  accessToken,
}) => {
  const [currentSSOConfig, setCurrentSSOConfig] = useState<SSOProviderConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [isViewModalVisible, setIsViewModalVisible] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  // Load current SSO configuration
  const loadSSOConfig = async () => {
    try {
      setLoading(true);
      const response = await getSSOProviderConfig(accessToken);
      setCurrentSSOConfig(response.config);
    } catch (error) {
      console.error("Failed to load SSO configuration:", error);
      // Don't show error message if SSO is just not configured yet
    } finally {
      setLoading(false);
    }
  };

  // Load SSO config when component mounts
  useEffect(() => {
    loadSSOConfig();
  }, [accessToken]);

  // Load existing values into form when editing
  useEffect(() => {
    if (isAddSSOModalVisible && currentSSOConfig && isEditing) {
      const formValues: Record<string, any> = {
        sso_provider: currentSSOConfig.sso_provider,
        proxy_base_url: currentSSOConfig.proxy_base_url,
        user_email: currentSSOConfig.user_email,
      };

      // Add provider-specific values
      if (currentSSOConfig.google) {
        formValues.google_client_id = currentSSOConfig.google.google_client_id;
        // Don't pre-fill secrets for security
      }
      if (currentSSOConfig.microsoft) {
        formValues.microsoft_client_id = currentSSOConfig.microsoft.microsoft_client_id;
        formValues.microsoft_tenant = currentSSOConfig.microsoft.microsoft_tenant;
      }
      if (currentSSOConfig.generic) {
        formValues.generic_client_id = currentSSOConfig.generic.generic_client_id;
        formValues.generic_authorization_endpoint = currentSSOConfig.generic.generic_authorization_endpoint;
        formValues.generic_token_endpoint = currentSSOConfig.generic.generic_token_endpoint;
        formValues.generic_userinfo_endpoint = currentSSOConfig.generic.generic_userinfo_endpoint;
        formValues.generic_scope = currentSSOConfig.generic.generic_scope;
      }

      form.setFieldsValue(formValues);
    }
  }, [isAddSSOModalVisible, currentSSOConfig, isEditing, form]);

  // Handle form submission
  const handleSSOSubmit = async (formValues: Record<string, any>) => {
    try {
      setLoading(true);
      const ssoConfig: SSOConfigRequest = {
        sso_provider: formValues.sso_provider,
        proxy_base_url: formValues.proxy_base_url,
        user_email: formValues.user_email,
        google_client_id: formValues.google_client_id,
        google_client_secret: formValues.google_client_secret,
        microsoft_client_id: formValues.microsoft_client_id,
        microsoft_client_secret: formValues.microsoft_client_secret,
        microsoft_tenant: formValues.microsoft_tenant,
        generic_client_id: formValues.generic_client_id,
        generic_client_secret: formValues.generic_client_secret,
        generic_authorization_endpoint: formValues.generic_authorization_endpoint,
        generic_token_endpoint: formValues.generic_token_endpoint,
        generic_userinfo_endpoint: formValues.generic_userinfo_endpoint,
        generic_scope: formValues.generic_scope || "openid email profile",
      };

      await updateSSOProviderConfig(accessToken, ssoConfig);
      await loadSSOConfig(); // Reload the configuration
      setIsEditing(false);
      handleShowInstructions(formValues);
    } catch (error) {
      console.error("Failed to update SSO configuration:", error);
      message.error("Failed to update SSO configuration");
    } finally {
      setLoading(false);
    }
  };

  // Handle SSO deletion
  const handleDeleteSSO = async () => {
    try {
      setLoading(true);
      await deleteSSOProviderConfig(accessToken);
      setCurrentSSOConfig(null);
      setIsViewModalVisible(false);
    } catch (error) {
      console.error("Failed to delete SSO configuration:", error);
      message.error("Failed to delete SSO configuration");
    } finally {
      setLoading(false);
    }
  };

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
        {field.name.includes('secret') ? (
          <Input.Password placeholder={isEditing ? "Leave blank to keep existing value" : field.placeholder} />
        ) : (
          <TextInput placeholder={field.placeholder} />
        )}
      </Form.Item>
    ));
  };

  // Helper function to render SSO configuration view
  const renderSSOConfigView = () => {
    if (!currentSSOConfig) return null;

    const provider = currentSSOConfig.sso_provider;
    if (!provider) return null;

    const providerConfig = currentSSOConfig[provider as keyof SSOProviderConfig];

    return (
      <div>
        <Card 
          title={
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>Current SSO Configuration</span>
              <div>
                <Button2 
                  icon={<EditOutlined />} 
                  onClick={() => {
                    setIsEditing(true);
                    setIsViewModalVisible(false);
                    // The modal will open in edit mode
                  }}
                  style={{ marginRight: 8 }}
                >
                  Edit
                </Button2>
                <Popconfirm
                  title="Delete SSO Configuration"
                  description="Are you sure you want to delete the SSO configuration? This action cannot be undone."
                  onConfirm={handleDeleteSSO}
                  okText="Yes"
                  cancelText="No"
                >
                  <Button2 danger icon={<DeleteOutlined />}>
                    Delete
                  </Button2>
                </Popconfirm>
              </div>
            </div>
          }
        >
          <div style={{ marginBottom: 16 }}>
            <Text><strong>Provider:</strong> {provider.charAt(0).toUpperCase() + provider.slice(1)} SSO</Text>
          </div>
          <div style={{ marginBottom: 16 }}>
            <Text><strong>Proxy Base URL:</strong> {currentSSOConfig.proxy_base_url}</Text>
          </div>
          <div style={{ marginBottom: 16 }}>
            <Text><strong>Admin Email:</strong> {currentSSOConfig.user_email}</Text>
          </div>
          
          {providerConfig && typeof providerConfig === 'object' && (
            <div>
              <Title level={5}>Provider Configuration:</Title>
              {Object.entries(providerConfig).map(([key, value]) => (
                <div key={key} style={{ marginBottom: 8 }}>
                  <Text>
                    <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong>{' '}
                    {key.includes('secret') ? '***' : String(value)}
                  </Text>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    );
  };

  return (
    <>
      {/* Existing SSO Configuration View */}
      {currentSSOConfig && !isAddSSOModalVisible && (
        <div style={{ marginBottom: 16 }}>
          {renderSSOConfigView()}
        </div>
      )}

      {/* Add/Edit SSO Modal */}
      <Modal
        title={isEditing ? "Edit SSO Configuration" : "Add SSO"}
        visible={isAddSSOModalVisible}
        width={800}
        footer={null}
        onOk={handleAddSSOOk}
        onCancel={() => {
          handleAddSSOCancel();
          setIsEditing(false);
        }}
      >
        <Spin spinning={loading}>
          <Form
            form={form}
            onFinish={handleSSOSubmit}
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
                <Select disabled={isEditing}>
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
              <Button2 htmlType="submit" loading={loading}>
                {isEditing ? "Update" : "Save"}
              </Button2>
            </div>
          </Form>
        </Spin>
      </Modal>

      {/* Instructions Modal */}
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

      {/* Add SSO Button when no configuration exists */}
      {!currentSSOConfig && !isAddSSOModalVisible && (
        <div style={{ textAlign: "center", padding: "40px" }}>
          <Title level={4}>No SSO Configuration Found</Title>
          <Paragraph>Configure SSO to enable single sign-on authentication for your users.</Paragraph>
          <Button2 
            type="primary" 
            onClick={() => {
              setIsEditing(false);
              // This should trigger the parent to show the add modal
            }}
          >
            Add SSO Configuration
          </Button2>
        </div>
      )}
    </>
  );
};

export { ssoProviderConfigs };  // Export for use in other components
export default SSOModals; 