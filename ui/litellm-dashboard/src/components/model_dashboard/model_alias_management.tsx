import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Button
} from "@tremor/react";
import {
  Input,
  Select,
  message,
  Button as AntdButton,
} from "antd";
import {
  PencilIcon,
  TrashIcon,
} from "@heroicons/react/outline";
import { getRouterSettings, updateRouterSettings } from "../networking";

interface ModelAliasManagementProps {
  accessToken: string;
  availableModels: string[];
  onRefresh: () => void;
}

const ModelAliasManagement: React.FC<ModelAliasManagementProps> = ({
  accessToken,
  availableModels,
  onRefresh,
}) => {
  const [modelGroupAliases, setModelGroupAliases] = useState<Record<string, string>>({});
  const [newAliasName, setNewAliasName] = useState<string>("");
  const [selectedModelForAlias, setSelectedModelForAlias] = useState<string>("");
  const [editingAlias, setEditingAlias] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch router settings
  useEffect(() => {
    fetchRouterSettings();
  }, [accessToken]);

  const fetchRouterSettings = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const settings = await getRouterSettings(accessToken);
      console.log("Router settings response:", settings); // Log the response for debugging
      
      // Check both possible structures for model_group_alias
      if (settings?.router_settings?.model_group_alias) {
        setModelGroupAliases(settings.router_settings.model_group_alias);
      } else if (settings?.model_group_alias) {
        // Direct structure as shown in user's example
        setModelGroupAliases(settings.model_group_alias);
      } else {
        setModelGroupAliases({});
      }
    } catch (error) {
      console.error("Failed to fetch router settings:", error);
      message.error("Failed to load model aliases");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle adding or updating a model group alias
  const handleSaveModelGroupAlias = async () => {
    if (!accessToken || !newAliasName || !selectedModelForAlias) {
      message.error("Alias name and model selection are required");
      return;
    }

    setIsLoading(true);
    try {
      const updatedAliases = { ...modelGroupAliases };
      
      // If editing, remove the old alias first
      if (editingAlias && editingAlias !== newAliasName) {
        delete updatedAliases[editingAlias];
      }
      
      updatedAliases[newAliasName] = selectedModelForAlias;
      
      await updateRouterSettings(accessToken, {
        model_group_alias: updatedAliases
      });
      
      setModelGroupAliases(updatedAliases);
      setNewAliasName("");
      setSelectedModelForAlias("");
      setEditingAlias(null);
      
      message.success(`Model group alias ${editingAlias ? 'updated' : 'created'} successfully`);
      onRefresh();
    } catch (error) {
      console.error("Failed to save model group alias:", error);
      message.error(`Failed to ${editingAlias ? 'update' : 'create'} model group alias`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle deleting a model group alias
  const handleDeleteModelGroupAlias = async (aliasName: string) => {
    setIsLoading(true);
    try {
      const updatedAliases = { ...modelGroupAliases };
      delete updatedAliases[aliasName];
      
      await updateRouterSettings(accessToken, {
        model_group_alias: updatedAliases
      });
      
      setModelGroupAliases(updatedAliases);
      message.success(`Model group alias deleted successfully`);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete model group alias:", error);
      message.error("Failed to delete model group alias");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle editing a model group alias
  const handleEditModelGroupAlias = (aliasName: string) => {
    setEditingAlias(aliasName);
    setNewAliasName(aliasName);
    setSelectedModelForAlias(modelGroupAliases[aliasName] || "");
  };

  return (
    <Card>
      <Title>Model Group Aliases</Title>
      <Text className="mb-6">
        Map model alias names to actual model names. Aliases can be used interchangeably with the original model names in API calls.
      </Text>
      
      <div className="mb-8">
        <form className="flex flex-col space-y-4">
          <div className="flex space-x-4 items-end">
            <div className="flex-1">
              <Text>Alias Name</Text>
              <Input 
                placeholder="Enter alias name" 
                value={newAliasName}
                onChange={(e) => setNewAliasName(e.target.value)}
              />
            </div>
            
            <div className="flex-1">
              <Text>Target Model</Text>
              <Select
                placeholder="Select model"
                value={selectedModelForAlias}
                onChange={(value: string) => setSelectedModelForAlias(value)}
                style={{ width: '100%' }}
              >
                {availableModels.map((model, idx) => (
                  <Select.Option key={idx} value={model}>
                    {model}
                  </Select.Option>
                ))}
              </Select>
            </div>
            
            <AntdButton 
              onClick={handleSaveModelGroupAlias}
              disabled={!newAliasName || !selectedModelForAlias || isLoading}
              type="primary"
              loading={isLoading}
            >
              {editingAlias ? "Update" : "Add"} Alias
            </AntdButton>
            
            {editingAlias && (
              <AntdButton 
                onClick={() => {
                  setEditingAlias(null);
                  setNewAliasName("");
                  setSelectedModelForAlias("");
                }}
                disabled={isLoading}
              >
                Cancel
              </AntdButton>
            )}
          </div>
        </form>
      </div>
      
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Alias Name</TableHeaderCell>
            <TableHeaderCell>Target Model</TableHeaderCell>
            <TableHeaderCell>Actions</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {Object.entries(modelGroupAliases).map(([alias, model], idx) => (
            <TableRow key={idx}>
              <TableCell>{alias}</TableCell>
              <TableCell>{model}</TableCell>
              <TableCell>
                <div className="flex space-x-2">
                  <Button
                    size="xs"
                    variant="secondary"
                    icon={PencilIcon}
                    onClick={() => handleEditModelGroupAlias(alias)}
                    disabled={isLoading}
                  />
                  <Button
                    size="xs"
                    variant="secondary"
                    icon={TrashIcon}
                    onClick={() => handleDeleteModelGroupAlias(alias)}
                    disabled={isLoading}
                  />
                </div>
              </TableCell>
            </TableRow>
          ))}
          {Object.keys(modelGroupAliases).length === 0 && (
            <TableRow>
              <TableCell colSpan={3} className="text-center">
                No model group aliases defined
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </Card>
  );
};

export default ModelAliasManagement; 