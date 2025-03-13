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
  Badge
} from "@tremor/react";
import { PlusIcon } from "@heroicons/react/solid";
import { getCredentialsList } from "./networking"; // Assume this is your networking function

interface CredentialsPanelProps {
  accessToken: string | null;
}

interface CredentialsResponse {
  credentials: CredentialItem[];
}

interface CredentialItem {
  credential_name: string | null;
  provider: string;
  credential_values: {
    api_key?: string;
    api_base?: string;
  };
  credential_info: {
    description?: string;
    type: string;
    required: boolean;
    default?: string;
  };
}

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ accessToken }) => {
  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    
    const fetchCredentials = async () => {
      try {
        const response: CredentialsResponse = await getCredentialsList(accessToken);
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
        <button 
          onClick={() => setIsAddModalOpen(true)}
          className="flex items-center bg-blue-500 text-white px-3 py-2 rounded-md hover:bg-blue-600 transition-colors"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          Add Credential
        </button>
      </div>

      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Credential Name</TableHeaderCell>
              <TableHeaderCell>Provider</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
              <TableHeaderCell>Required</TableHeaderCell>
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
                    {renderProviderBadge(credential.provider)}
                  </TableCell>
                  <TableCell>{credential.credential_info.description || '-'}</TableCell>
                  <TableCell>
                    <div className={`inline-flex rounded-full px-2 py-1 text-xs font-medium
                      ${credential.credential_info.required 
                        ? 'bg-green-100 text-green-800'  // Required styling
                        : 'bg-gray-100 text-gray-800'    // Optional styling
                      }`}>
                      {credential.credential_info.required ? 'Required' : 'Optional'}
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      {/* TODO: Implement Add Credential Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          {/* Modal content for adding credentials */}
        </div>
      )}
    </div>
  );
};

export default CredentialsPanel;