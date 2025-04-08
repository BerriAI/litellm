import React, { useState, useEffect } from "react";
import {
  Card,
  Text,
  Title,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Button,
  Badge,
} from "@tremor/react";
import {
  Form,
  Input,
  Select as Select2,
  message,
  Tooltip,
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { fetchUserModels } from "../create_key_button";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { tagInfoCall, tagUpdateCall } from "../networking";
import { Tag, TagInfoResponse } from "./types";

interface TagInfoViewProps {
  tagId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editTag: boolean;
}

const TagInfoView: React.FC<TagInfoViewProps> = ({
  tagId,
  onClose,
  accessToken,
  is_admin,
  editTag,
}) => {
  const [form] = Form.useForm();
  const [tagDetails, setTagDetails] = useState<Tag | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editTag);
  const [userModels, setUserModels] = useState<string[]>([]);

  const fetchTagDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await tagInfoCall(accessToken, [tagId]);
      const tagData = response[tagId];
      if (tagData) {
        setTagDetails(tagData);
        if (editTag) {
          form.setFieldsValue({
            name: tagData.name,
            description: tagData.description,
            models: tagData.models,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching tag details:", error);
      message.error("Error fetching tag details: " + error);
    }
  };

  useEffect(() => {
    fetchTagDetails();
  }, [tagId, accessToken]);

  useEffect(() => {
    if (accessToken) {
      // Using dummy values for userID and userRole since they're required by the function
      // TODO: Pass these as props if needed for the actual API implementation
      fetchUserModels("dummy-user", "Admin", accessToken, setUserModels);
    }
  }, [accessToken]);

  const handleSave = async (values: any) => {
    if (!accessToken) return;
    try {
      await tagUpdateCall(accessToken, {
        name: values.name,
        description: values.description,
        models: values.models,
      });
      message.success("Tag updated successfully");
      setIsEditing(false);
      fetchTagDetails();
    } catch (error) {
      console.error("Error updating tag:", error);
      message.error("Error updating tag: " + error);
    }
  };

  if (!tagDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <Button onClick={onClose}>← Back to Tags</Button>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>Edit Tag</Button>
        )}
      </div>

      <TabGroup>
        <TabList>
          <Tab>Overview</Tab>
          <Tab>Usage & Analytics</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {isEditing ? (
              <Card>
                <Form
                  form={form}
                  onFinish={handleSave}
                  layout="vertical"
                  initialValues={tagDetails}
                >
                  <Form.Item
                    label="Tag Name"
                    name="name"
                    rules={[{ required: true, message: "Please input a tag name" }]}
                  >
                    <Input />
                  </Form.Item>

                  <Form.Item
                    label="Description"
                    name="description"
                  >
                    <Input.TextArea rows={4} />
                  </Form.Item>

                  <Form.Item
                    label={
                      <span>
                        Allowed LLMs{' '}
                        <Tooltip title="Select which LLMs are allowed to process this type of data">
                          <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                        </Tooltip>
                      </span>
                    }
                    name="models"
                  >
                    <Select2
                      mode="multiple"
                      placeholder="Select LLMs"
                    >
                      {userModels.map((model) => (
                        <Select2.Option key={model} value={model}>
                          {getModelDisplayName(model)}
                        </Select2.Option>
                      ))}
                    </Select2>
                  </Form.Item>

                  <div className="flex justify-end space-x-2">
                    <Button onClick={() => setIsEditing(false)}>Cancel</Button>
                    <Button type="submit" color="blue">Save Changes</Button>
                  </div>
                </Form>
              </Card>
            ) : (
              <div className="space-y-6">
                <Card>
                  <Title>Tag Details</Title>
                  <div className="space-y-4 mt-4">
                    <div>
                      <Text className="font-medium">Name</Text>
                      <Text>{tagDetails.name}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Description</Text>
                      <Text>{tagDetails.description || "-"}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Allowed LLMs</Text>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {tagDetails.models.map((llm) => (
                          <Badge key={llm} color="blue">
                            {llm}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Created</Text>
                      <Text>{tagDetails.created_at ? new Date(tagDetails.created_at).toLocaleString() : "-"}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Last Updated</Text>
                      <Text>{tagDetails.updated_at ? new Date(tagDetails.updated_at).toLocaleString() : "-"}</Text>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </TabPanel>

          <TabPanel>
            <Card>
              <Title>Usage Statistics</Title>
              <div className="space-y-4 mt-4">
                <Text>Coming soon...</Text>
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default TagInfoView; 