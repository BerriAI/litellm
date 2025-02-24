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
              label="Admin Email"
              name="user_email"
              rules={[
                {
                  required: true,
                  message: "Please enter the email of the proxy admin",
                },
              ]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item
              label="PROXY BASE URL"
              name="proxy_base_url"
              rules={[
                {
                  required: true,
                  message: "Please enter the proxy base url",
                },
              ]}
            >
              <TextInput />
            </Form.Item>

            <Form.Item
              label="SSO Provider"
              name="sso_provider"
              rules={[
                {
                  required: true,
                  message: "Please select an SSO provider",
                },
              ]}
            >
              <Select>
                <Select.Option value="google">Google SSO</Select.Option>
                <Select.Option value="microsoft">Microsoft SSO</Select.Option>
                <Select.Option value="okta">Okta SSO</Select.Option>
                <Select.Option value="generic">Generic SSO Provider</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) => prevValues.sso_provider !== currentValues.sso_provider}
            >
              {({ getFieldValue }) => {
                const provider = getFieldValue('sso_provider');
                
                if (provider === 'google') {
                  return (
                    <>
                      <Form.Item
                        label="GOOGLE CLIENT ID"
                        name="google_client_id"
                        rules={[{ required: true, message: "Please enter the google client id" }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="GOOGLE CLIENT SECRET"
                        name="google_client_secret"
                        rules={[{ required: true, message: "Please enter the google client secret" }]}
                      >
                        <Input.Password />
                      </Form.Item>
                    </>
                  );
                }
                
                if (provider === 'microsoft') {
                  return (
                    <>
                      <Form.Item
                        label="MICROSOFT CLIENT ID"
                        name="microsoft_client_id"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="MICROSOFT CLIENT SECRET"
                        name="microsoft_client_secret"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="MICROSOFT TENANT"
                        name="microsoft_tenant"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                    </>
                  );
                }

                if (provider === 'okta') {
                  return (
                    <>
                      <Form.Item
                        label="GENERIC CLIENT ID"
                        name="generic_client_id"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="GENERIC CLIENT SECRET"
                        name="generic_client_secret"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="AUTHORIZATION ENDPOINT"
                        name="generic_authorization_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput placeholder="https://your-okta-domain/authorize" />
                      </Form.Item>
                      <Form.Item
                        label="TOKEN ENDPOINT"
                        name="generic_token_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput placeholder="https://your-okta-domain/token" />
                      </Form.Item>
                      <Form.Item
                        label="USERINFO ENDPOINT"
                        name="generic_userinfo_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput placeholder="https://your-okta-domain/userinfo" />
                      </Form.Item>
                    </>
                  );
                }

                if (provider === 'generic') {
                  return (
                    <>
                      <Form.Item
                        label="GENERIC CLIENT ID"
                        name="generic_client_id"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="GENERIC CLIENT SECRET"
                        name="generic_client_secret"
                        rules={[{ required: true }]}
                      >
                        <Input.Password />
                      </Form.Item>
                      <Form.Item
                        label="AUTHORIZATION ENDPOINT"
                        name="generic_authorization_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput />
                      </Form.Item>
                      <Form.Item
                        label="TOKEN ENDPOINT"
                        name="generic_token_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput />
                      </Form.Item>
                      <Form.Item
                        label="USERINFO ENDPOINT"
                        name="generic_userinfo_endpoint"
                        rules={[{ required: true }]}
                      >
                        <TextInput />
                      </Form.Item>
                    </>
                  );
                }
              }}
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

export default SSOModals; 