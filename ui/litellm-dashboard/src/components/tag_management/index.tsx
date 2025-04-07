import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Icon,
  Button,
  Badge,
  Col,
  Text,
  Grid,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Tab,
  TextInput,
} from "@tremor/react";
import {
  InformationCircleIcon,
  PencilAltIcon,
  TrashIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import {
  Modal,
  Form,
  Input,
  Select as Select2,
  message,
  Tooltip
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import NumericalInput from "../shared/numerical_input";
import TagInfoView from "./tag_info";

interface TagProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

interface Tag {
  name: string;
  description: string;
  severity: "Low" | "Medium" | "High";
  allowed_llms: string[];
  spend: number;
  created_at: string;
}

const TagManagement: React.FC<TagProps> = ({
  accessToken,
  userID,
  userRole,
}) => {
  const [tags, setTags] = useState<Tag[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [selectedTagId, setSelectedTagId] = useState<string | null>(null);
  const [editTag, setEditTag] = useState<boolean>(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [tagToDelete, setTagToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [form] = Form.useForm();

  const handleRefreshClick = () => {
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleCreate = async (formValues: any) => {
    try {
      // TODO: Implement tag creation API call
      message.success("Tag created successfully");
      setIsCreateModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error("Error creating tag:", error);
      message.error("Error creating tag: " + error);
    }
  };

  const handleDelete = async (tagName: string) => {
    setTagToDelete(tagName);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    try {
      // TODO: Implement tag deletion API call
      message.success("Tag deleted successfully");
    } catch (error) {
      console.error("Error deleting tag:", error);
      message.error("Error deleting tag: " + error);
    }
    setIsDeleteModalOpen(false);
    setTagToDelete(null);
  };

  // Mock data for demonstration
  const mockTags: Tag[] = [
    {
      name: "PII",
      description: "Personally Identifiable Information",
      severity: "High",
      allowed_llms: ["Claude 3 Opus", "GPT-4"],
      spend: 120.50,
      created_at: "2024-03-15",
    },
    {
      name: "Financial",
      description: "Financial data and transactions",
      severity: "High",
      allowed_llms: ["Claude 3 Opus"],
      spend: 85.75,
      created_at: "2024-03-14",
    }
  ];

  useEffect(() => {
    setTags(mockTags);
  }, [lastRefreshed]);

  return (
    <div className="w-full mx-4 h-[75vh]">
      {selectedTagId ? (
        <TagInfoView
          tagId={selectedTagId}
          onClose={() => {
            setSelectedTagId(null);
            setEditTag(false);
          }}
          accessToken={accessToken}
          is_admin={userRole === "Admin"}
          editTag={editTag}
        />
      ) : (
        <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
          <TabList className="flex justify-between mt-2 w-full items-center">
            <div className="flex">
              <Tab>Data Sensitivity Tags</Tab>
            </div>
            <div className="flex items-center space-x-2">
              {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
              <Icon
                icon={RefreshIcon}
                variant="shadow"
                size="xs"
                className="self-center"
                onClick={handleRefreshClick}
              />
            </div>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Text>
                Click on a tag name to view and edit its details.
              </Text>
              <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                <Col numColSpan={1}>
                  <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableHeaderCell>Tag Name</TableHeaderCell>
                          <TableHeaderCell>Description</TableHeaderCell>
                          <TableHeaderCell>Severity</TableHeaderCell>
                          <TableHeaderCell>Allowed LLMs</TableHeaderCell>
                          <TableHeaderCell>Spend (USD)</TableHeaderCell>
                          <TableHeaderCell>Created</TableHeaderCell>
                          <TableHeaderCell>Actions</TableHeaderCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {tags.map((tag) => (
                          <TableRow key={tag.name}>
                            <TableCell>
                              <Button
                                size="xs"
                                variant="light"
                                className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5"
                                onClick={() => setSelectedTagId(tag.name)}
                              >
                                {tag.name}
                              </Button>
                            </TableCell>
                            <TableCell>{tag.description}</TableCell>
                            <TableCell>
                              <Badge
                                color={
                                  tag.severity === "High"
                                    ? "red"
                                    : tag.severity === "Medium"
                                    ? "yellow"
                                    : "green"
                                }
                              >
                                {tag.severity}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1">
                                {tag.allowed_llms.map((llm) => (
                                  <Badge
                                    key={llm}
                                    size="xs"
                                    className="mb-1"
                                    color="blue"
                                  >
                                    {llm}
                                  </Badge>
                                ))}
                              </div>
                            </TableCell>
                            <TableCell>${tag.spend.toFixed(2)}</TableCell>
                            <TableCell>{new Date(tag.created_at).toLocaleDateString()}</TableCell>
                            <TableCell>
                              <div className="flex space-x-2">
                                <Icon
                                  icon={PencilAltIcon}
                                  size="sm"
                                  onClick={() => {
                                    setSelectedTagId(tag.name);
                                    setEditTag(true);
                                  }}
                                />
                                <Icon
                                  onClick={() => handleDelete(tag.name)}
                                  icon={TrashIcon}
                                  size="sm"
                                />
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Card>
                </Col>
                {userRole === "Admin" && (
                  <Col numColSpan={1}>
                    <Button
                      className="mx-auto"
                      onClick={() => setIsCreateModalVisible(true)}
                    >
                      + Create New Tag
                    </Button>
                  </Col>
                )}
              </Grid>

              {/* Create Tag Modal */}
              <Modal
                title="Create New Tag"
                visible={isCreateModalVisible}
                width={800}
                footer={null}
                onCancel={() => {
                  setIsCreateModalVisible(false);
                  form.resetFields();
                }}
              >
                <Form
                  form={form}
                  onFinish={handleCreate}
                  labelCol={{ span: 8 }}
                  wrapperCol={{ span: 16 }}
                  labelAlign="left"
                >
                  <Form.Item
                    label="Tag Name"
                    name="tag_name"
                    rules={[{ required: true, message: "Please input a tag name" }]}
                  >
                    <TextInput placeholder="e.g., PII, Financial, Medical" />
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

                  <div style={{ textAlign: "right", marginTop: "10px" }}>
                    <Button type="submit">
                      Create Tag
                    </Button>
                  </div>
                </Form>
              </Modal>

              {/* Delete Confirmation Modal */}
              {isDeleteModalOpen && (
                <div className="fixed z-10 inset-0 overflow-y-auto">
                  <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                    <div className="fixed inset-0 transition-opacity" aria-hidden="true">
                      <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
                    </div>
                    <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                      <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                        <div className="sm:flex sm:items-start">
                          <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                            <h3 className="text-lg leading-6 font-medium text-gray-900">
                              Delete Tag
                            </h3>
                            <div className="mt-2">
                              <p className="text-sm text-gray-500">
                                Are you sure you want to delete this tag?
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                        <Button
                          onClick={confirmDelete}
                          color="red"
                          className="ml-2"
                        >
                          Delete
                        </Button>
                        <Button onClick={() => {
                          setIsDeleteModalOpen(false);
                          setTagToDelete(null);
                        }}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </TabPanel>
          </TabPanels>
        </TabGroup>
      )}
    </div>
  );
};

export default TagManagement;
