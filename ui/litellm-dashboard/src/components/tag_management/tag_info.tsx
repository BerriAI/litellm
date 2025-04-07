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

interface TagInfoViewProps {
  tagId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editTag: boolean;
}

interface TagDetails {
  name: string;
  description: string;
  severity: "Low" | "Medium" | "High";
  allowed_llms: string[];
  spend: number;
  created_at: string;
  updated_at: string;
}

const TagInfoView: React.FC<TagInfoViewProps> = ({
  tagId,
  onClose,
  accessToken,
  is_admin,
  editTag,
}) => {
  const [form] = Form.useForm();
  const [tagDetails, setTagDetails] = useState<TagDetails | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editTag);

  // Mock data for demonstration
  const mockTagDetails: TagDetails = {
    name: "PII",
    description: "Personally Identifiable Information",
    severity: "High",
    allowed_llms: ["Claude 3 Opus", "GPT-4"],
    spend: 120.50,
    created_at: "2024-03-15",
    updated_at: "2024-03-15",
  };

  useEffect(() => {
    // TODO: Implement API call to fetch tag details
    setTagDetails(mockTagDetails);
    if (editTag) {
      form.setFieldsValue(mockTagDetails);
    }
  }, [tagId]);

  const handleSave = async (values: any) => {
    try {
      // TODO: Implement API call to update tag
      message.success("Tag updated successfully");
      setIsEditing(false);
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
                    rules={[{ required: true, message: "Please provide a description" }]}
                  >
                    <Input.TextArea rows={4} />
                  </Form.Item>

                  <Form.Item
                    label="Severity"
                    name="severity"
                    rules={[{ required: true, message: "Please select severity" }]}
                  >
                    <Select2>
                      <Select2.Option value="Low">Low</Select2.Option>
                      <Select2.Option value="Medium">Medium</Select2.Option>
                      <Select2.Option value="High">High</Select2.Option>
                    </Select2>
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
                    name="allowed_llms"
                  >
                    <Select2
                      mode="multiple"
                      placeholder="Select LLMs"
                    >
                      <Select2.Option value="gpt-4">GPT-4</Select2.Option>
                      <Select2.Option value="claude-3-opus">Claude 3 Opus</Select2.Option>
                      <Select2.Option value="claude-3-sonnet">Claude 3 Sonnet</Select2.Option>
                      <Select2.Option value="claude-3-haiku">Claude 3 Haiku</Select2.Option>
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
                      <Text>{tagDetails.description}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Severity</Text>
                      <Badge
                        color={
                          tagDetails.severity === "High"
                            ? "red"
                            : tagDetails.severity === "Medium"
                            ? "yellow"
                            : "green"
                        }
                      >
                        {tagDetails.severity}
                      </Badge>
                    </div>
                    <div>
                      <Text className="font-medium">Allowed LLMs</Text>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {tagDetails.allowed_llms.map((llm) => (
                          <Badge key={llm} color="blue">
                            {llm}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Created</Text>
                      <Text>{new Date(tagDetails.created_at).toLocaleString()}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Last Updated</Text>
                      <Text>{new Date(tagDetails.updated_at).toLocaleString()}</Text>
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
                <div>
                  <Text className="font-medium">Total Spend</Text>
                  <Text>${tagDetails.spend.toFixed(2)} USD</Text>
                </div>
                {/* Add more analytics components here */}
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default TagInfoView; 