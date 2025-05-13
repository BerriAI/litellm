import React, { useState, useEffect } from "react";
import {
  Card,
  Icon,
  Button,
  Col,
  Text,
  Grid,
  TextInput,
} from "@tremor/react";
import {
  InformationCircleIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import {
  Modal,
  Form,
  Select as Select2,
  message,
  Tooltip,
  Input
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import NumericalInput from "../shared/numerical_input";
import TagInfoView from "./tag_info";
import { modelInfoCall } from "../networking";
import { tagCreateCall, tagListCall, tagDeleteCall } from "../networking";
import { Tag } from "./types";
import TagTable from "./TagTable";

interface ModelInfo {
  model_name: string;
  litellm_params: {
    model: string;
  };
  model_info: {
    id: string;
  };
}

interface TagProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
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
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);

  const fetchTags = async () => {
    if (!accessToken) return;
    try {
      const response = await tagListCall(accessToken);
      console.log("List tags response:", response);
      setTags(Object.values(response));
    } catch (error) {
      console.error("Error fetching tags:", error);
      message.error("Error fetching tags: " + error);
    }
  };

  const handleRefreshClick = () => {
    fetchTags();
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleCreate = async (formValues: any) => {
    if (!accessToken) return;
    try {
      await tagCreateCall(accessToken, {
        name: formValues.tag_name,
        description: formValues.description,
        models: formValues.allowed_llms,
      });
      message.success("Tag created successfully");
      setIsCreateModalVisible(false);
      form.resetFields();
      fetchTags();
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
    if (!accessToken || !tagToDelete) return;
    try {
      await tagDeleteCall(accessToken, tagToDelete);
      message.success("Tag deleted successfully");
      fetchTags();
    } catch (error) {
      console.error("Error deleting tag:", error);
      message.error("Error deleting tag: " + error);
    }
    setIsDeleteModalOpen(false);
    setTagToDelete(null);
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      const fetchModels = async () => {
        try {
          const response = await modelInfoCall(accessToken, userID, userRole);
          if (response && response.data) {
            setAvailableModels(response.data);
          }
        } catch (error) {
          console.error("Error fetching models:", error);
          message.error("Error fetching models: " + error);
        }
      };
      fetchModels();
    }
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    fetchTags();
  }, [accessToken]);

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
        <div className="gap-2 p-8 h-[75vh] w-full mt-2">
          <div className="flex justify-between mt-2 w-full items-center mb-4">
            <h1>Tag Management</h1>
            <div className="flex items-center space-x-2">
              {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
              <Icon
                icon={RefreshIcon}
                variant="shadow"
                size="xs"
                className="self-center cursor-pointer"
                onClick={handleRefreshClick}
              />
            </div>
          </div>
          
          
          <Text className="mb-4">
            Click on a tag name to view and edit its details.

            <p>You can use tags to restrict the usage of certain LLMs based on tags passed in the request. Read more about tag routing <a href="https://docs.litellm.ai/docs/proxy/tag_routing" target="_blank" rel="noopener noreferrer">here</a>.</p>
          </Text>

          <Button
            className="mb-4"
            onClick={() => setIsCreateModalVisible(true)}
          >
            + Create New Tag
          </Button>

          <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
            <Col numColSpan={1}>
              <TagTable
                data={tags}
                onEdit={(tag) => {
                  setSelectedTagId(tag.name);
                  setEditTag(true);
                }}
                onDelete={handleDelete}
                onSelectTag={setSelectedTagId}
              />
            </Col>
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
                <TextInput />
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
                    Allowed Models{' '}
                    <Tooltip title="Select which LLMs are allowed to process requests from this tag">
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
                  {availableModels.map((model) => (
                    <Select2.Option key={model.model_info.id} value={model.model_info.id}>
                      <div>
                        <span>{model.model_name}</span>
                        <span className="text-gray-400 ml-2">({model.model_info.id})</span>
                      </div>
                    </Select2.Option>
                  ))}
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
        </div>
      )}
    </div>
  );
};

export default TagManagement;
