import React, { useState } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Col,
  Button as TremorButton,
  Callout,
} from "@tremor/react";
import { Button, message, Form, Input } from "antd";
import { keyCreateCall } from "./networking";
import { CopyToClipboard } from "react-copy-to-clipboard";

interface SCIMConfigProps {
  accessToken: string | null;
  userID: string | null;
  baseUrl: string;
}

const SCIMConfig: React.FC<SCIMConfigProps> = ({ accessToken, userID, baseUrl }) => {
  const [form] = Form.useForm();
  const [isCreatingToken, setIsCreatingToken] = useState(false);
  const [tokenData, setTokenData] = useState<any>(null);
  
  const scimBaseUrl = `${baseUrl}/scim`;
  
  const handleCreateSCIMToken = async (values: any) => {
    if (!accessToken || !userID) {
      message.error("You need to be logged in to create a SCIM token");
      return;
    }
    
    try {
      setIsCreatingToken(true);
      
      const formData = {
        key_alias: values.key_alias || "SCIM Access Token",
        team_id: null,
        models: [],
        allowed_routes: ["/scim/*"],
      };
      
      const response = await keyCreateCall(accessToken, userID, formData);
      setTokenData(response);
      message.success("SCIM token created successfully");
    } catch (error) {
      console.error("Error creating SCIM token:", error);
      message.error("Failed to create SCIM token");
    } finally {
      setIsCreatingToken(false);
    }
  };

  return (
    <Grid numItems={1}>
      <Card>
        <Title className="mb-4">SCIM Configuration</Title>
        <Text>
          System for Cross-domain Identity Management (SCIM) allows you to automatically provision and manage users and groups in LiteLLM.
        </Text>
        
        <div className="mt-6">
          <Title className="text-lg">SCIM Base URL</Title>
          <div className="flex items-center mt-2 mb-6">
            <Input
              value={scimBaseUrl}
              readOnly
              className="flex-grow"
            />
            <CopyToClipboard
              text={scimBaseUrl}
              onCopy={() => message.success("URL copied to clipboard")}
            >
              <Button type="primary" className="ml-2">
                Copy
              </Button>
            </CopyToClipboard>
          </div>

          <Callout title="Using SCIM" color="blue" className="mt-4 mb-6">
            You need a SCIM token to authenticate with the SCIM API. Create one below and use it in your SCIM provider's configuration.
          </Callout>

          {!tokenData ? (
            <>
              <Title className="text-lg mb-4">Create SCIM Access Token</Title>
              <Form
                form={form}
                onFinish={handleCreateSCIMToken}
                layout="vertical"
              >
                <Form.Item
                  name="key_alias"
                  label="Token Name"
                  rules={[{ required: true, message: "Please enter a name for your token" }]}
                >
                  <Input placeholder="SCIM Access Token" />
                </Form.Item>
                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={isCreatingToken}
                  >
                    Create SCIM Token
                  </Button>
                </Form.Item>
              </Form>
            </>
          ) : (
            <Card className="mt-4 bg-yellow-50">
              <Title className="text-lg text-yellow-800">Your SCIM Token</Title>
              <Text className="text-yellow-800 mb-2">
                Make sure to copy this token now. You won't be able to see it again!
              </Text>
              <div className="flex items-center mt-2">
                <Input.Password
                  value={tokenData.token}
                  className="flex-grow mr-2"
                />
                <CopyToClipboard
                  text={tokenData.token}
                  onCopy={() => message.success("Token copied to clipboard")}
                >
                  <Button type="primary">
                    Copy
                  </Button>
                </CopyToClipboard>
              </div>
              <TremorButton
                className="mt-4"
                onClick={() => setTokenData(null)}
              >
                Create Another Token
              </TremorButton>
            </Card>
          )}
        </div>
      </Card>
    </Grid>
  );
};

export default SCIMConfig; 