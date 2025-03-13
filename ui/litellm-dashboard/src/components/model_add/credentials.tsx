import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Text,
  Badge,
  Button
} from "@tremor/react";
import { UploadProps } from "antd/es/upload";
import { PlusIcon } from "@heroicons/react/solid";
import { credentialListCall, credentialCreateCall } from "@/components/networking"; // Assume this is your networking function
import AddCredentialsTab from "./add_credentials_tab";
import { Form, message } from "antd";
interface CredentialsPanelProps {
  accessToken: string | null;
  uploadProps: UploadProps;
}

interface CredentialsResponse {
  credentials: CredentialItem[];
}

interface CredentialItem {
  credential_name: string | null;
  credential_values: object;
  credential_info: {
    custom_llm_provider?: string;
    description?: string;
    required?: boolean;
  };
}

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ accessToken, uploadProps }) => {
  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [form] = Form.useForm();

  const handleAddCredential = async (values: any) => {
    if (!accessToken) {
      console.error('No access token found');
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !['credential_name', 'custom_llm_provider'].includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    // Transform form values into credential structure
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      }
    };

    // Add to list and close modal
    const response = await credentialCreateCall(accessToken, newCredential);
    message.success('Credential added successfully');
    console.log(`response: ${JSON.stringify(response)}`);
    setIsAddModalOpen(false);
    form.resetFields();
  };

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    
    const fetchCredentials = async () => {
      try {
        const response: CredentialsResponse = await credentialListCall(accessToken);
        console.log(`credentials: ${JSON.stringify(response)}`);
        setCredentialsList(response.credentials);
      } catch (error) {
        console.error('Error fetching credentials:', error);
      }
    };
    fetchCredentials();
  }, [accessToken]);

  const renderProviderBadge = (provider: string) => {
    const providerColors: Record<string, string> = {
      'openai': 'blue',
      'azure': 'indigo',
      'anthropic': 'purple',
      'default': 'gray'
    };

    const color = providerColors[provider.toLowerCase()] || providerColors['default'];
    return (
      <Badge color={color as any} size="xs">
        {provider}
      </Badge>
    );
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex justify-between items-center mb-4">
        <Text>
          Configured credentials for different AI providers. Add and manage your API credentials.{" "}
          <a 
            href="https://docs.litellm.ai/docs/credentials" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-blue-500 hover:text-blue-700 underline"
          >
            Docs
          </a>
        </Text>
      </div>

      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Credential Name</TableHeaderCell>
              <TableHeaderCell>Provider</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(!credentialsList || credentialsList.length === 0) ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4 text-gray-500">
                  No credentials configured
                </TableCell>
              </TableRow>
            ) : (
              credentialsList.map((credential: CredentialItem, index: number) => (
                <TableRow key={index}>
                  <TableCell>{credential.credential_name}</TableCell>
                  <TableCell>
                    {renderProviderBadge(credential.credential_info?.custom_llm_provider as string || '-')}
                  </TableCell>
                  <TableCell>{credential.credential_info?.description || '-'}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>
      <Button 
        onClick={() => setIsAddModalOpen(true)}
        className="mt-4"
      >
        Add Credential
      </Button>

      

      {isAddModalOpen && (
        <AddCredentialsTab 
          form={form}
          onAddCredential={handleAddCredential}
          isVisible={isAddModalOpen}
          onCancel={() => setIsAddModalOpen(false)}
          uploadProps={uploadProps}
        />
      )}
    </div>
  );
};

export default CredentialsPanel;