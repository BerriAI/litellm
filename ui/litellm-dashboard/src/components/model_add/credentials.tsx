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
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  RefreshIcon,
  StatusOnlineIcon,
  TrashIcon,
} from "@heroicons/react/outline";
import { UploadProps } from "antd/es/upload";
import { PlusIcon } from "@heroicons/react/solid";
import { credentialListCall, credentialCreateCall, credentialDeleteCall, credentialUpdateCall, CredentialItem, CredentialsResponse } from "@/components/networking"; // Assume this is your networking function
import AddCredentialsTab from "./add_credentials_tab";
import { Form, message } from "antd";
interface CredentialsPanelProps {
  accessToken: string | null;
  uploadProps: UploadProps;
  credentialList: CredentialItem[];
  fetchCredentials: (accessToken: string) => Promise<void>;
}



const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ accessToken, uploadProps, credentialList, fetchCredentials }) => {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<CredentialItem | null>(null);
  const [form] = Form.useForm();
  console.log(`selectedCredential in credentials panel: ${JSON.stringify(selectedCredential)}`);

  const restrictedFields = ['credential_name', 'custom_llm_provider'];
  const handleUpdateCredential = async (values: any) => {
    if (!accessToken) {
      console.error('No access token found');
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    // Transform form values into credential structure
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      }
    };

    const response = await credentialUpdateCall(accessToken, values.credential_name, newCredential);
    message.success('Credential updated successfully');
    console.log(`response: ${JSON.stringify(response)}`);
    setIsUpdateModalOpen(false);
    fetchCredentials(accessToken);
  }

  const handleAddCredential = async (values: any) => {
    if (!accessToken) {
      console.error('No access token found');
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
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
    fetchCredentials(accessToken);
  };

  

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    fetchCredentials(accessToken);
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


  const handleDeleteCredential = async (credentialName: string) => {
    if (!accessToken) {
      console.error('No access token found');
      return;
    }
    const response = await credentialDeleteCall(accessToken, credentialName);
    console.log(`response: ${JSON.stringify(response)}`);
    message.success('Credential deleted successfully');
    fetchCredentials(accessToken);
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
            {(!credentialList || credentialList.length === 0) ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4 text-gray-500">
                  No credentials configured
                </TableCell>
              </TableRow>
            ) : (
              credentialList.map((credential: CredentialItem, index: number) => (
                <TableRow key={index}>
                  <TableCell>{credential.credential_name}</TableCell>
                  <TableCell>
                    {renderProviderBadge(credential.credential_info?.custom_llm_provider as string || '-')}
                  </TableCell>
                  <TableCell>{credential.credential_info?.description || '-'}</TableCell>
                  <TableCell>
                    <Button
                      icon={PencilAltIcon}
                      variant="light" 
                      size="sm"
                      onClick={() => {
                        console.log(`credential being set: ${JSON.stringify(credential)}`);
                        setSelectedCredential(credential);
                        setIsUpdateModalOpen(true);
                      }}
                    />
                    <Button
                      icon={TrashIcon}
                      variant="light"
                      size="sm"
                      onClick={() => handleDeleteCredential(credential.credential_name)}
                    />
                  </TableCell>
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
          onAddCredential={handleAddCredential}
          isVisible={isAddModalOpen}
          onCancel={() => setIsAddModalOpen(false)}
          uploadProps={uploadProps}
          addOrEdit="add"
          onUpdateCredential={handleUpdateCredential}
          existingCredential={null}
        />
      )}
      {isUpdateModalOpen && (
        <AddCredentialsTab 
          onAddCredential={handleAddCredential}
          isVisible={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onUpdateCredential={handleUpdateCredential}
          uploadProps={uploadProps}
          onCancel={() => setIsUpdateModalOpen(false)}
          addOrEdit="edit"
        />
      )}
    </div>
  );
};

export default CredentialsPanel;