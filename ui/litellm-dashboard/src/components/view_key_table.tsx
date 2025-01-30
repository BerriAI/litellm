"use client";
import React, { useEffect, useState } from "react";
import {
  keyDeleteCall,
  modelAvailableCall,
  getGuardrailsList,
} from "./networking";
import { add } from "date-fns";
import {
  InformationCircleIcon,
  StatusOnlineIcon,
  TrashIcon,
  PencilAltIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import {
  keySpendLogsCall,
  PredictedSpendLogsCall,
  keyUpdateCall,
  modelInfoCall,
  regenerateKeyCall,
} from "./networking";
import {
  Badge,
  Card,
  Table,
  Grid,
  Col,
  Button,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Dialog,
  DialogPanel,
  Text,
  Title,
  Subtitle,
  Icon,
  BarChart,
  TextInput,
  Textarea,
} from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import {
  fetchAvailableModelsForTeamOrKey,
  getModelDisplayName,
} from "./key_team_helpers/fetch_available_models_team_key";
import {
  Select as Select3,
  SelectItem,
  MultiSelect,
  MultiSelectItem,
} from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
  Select,
  Tooltip,
  DatePicker,
} from "antd";

import { CopyToClipboard } from "react-copy-to-clipboard";
import TextArea from "antd/es/input/TextArea";

const { Option } = Select;
const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function () {};
}

interface EditKeyModalProps {
  visible: boolean;
  onCancel: () => void;
  token: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

interface ModelLimitModalProps {
  visible: boolean;
  onCancel: () => void;
  token: ItemData;
  onSubmit: (updatedMetadata: any) => void;
  accessToken: string;
}

// Define the props type
interface ViewKeyTableProps {
  userID: string;
  userRole: string | null;
  accessToken: string;
  selectedTeam: any | null;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
  teams: any[] | null;
  premiumUser: boolean;
}

interface ItemData {
  key_alias: string | null;
  key_name: string;
  spend: string;
  max_budget: string | null;
  models: string[];
  tpm_limit: string | null;
  rpm_limit: string | null;
  token: string;
  token_id: string | null;
  id: number;
  team_id: string;
  metadata: any;
  user_id: string | null;
  expires: any;
  budget_duration: string | null;
  budget_reset_at: string | null;
  // Add any other properties that exist in the item data
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  userRole,
  accessToken,
  selectedTeam,
  data,
  setData,
  teams,
  premiumUser,
}) => {
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<ItemData | null>(null);
  const [spendData, setSpendData] = useState<
    { day: string; spend: number }[] | null
  >(null);
  const [predictedSpendString, setPredictedSpendString] = useState("");

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [infoDialogVisible, setInfoDialogVisible] = useState(false);
  const [selectedToken, setSelectedToken] = useState<ItemData | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const initialKnownTeamIDs: Set<string> = new Set();
  const [modelLimitModalVisible, setModelLimitModalVisible] = useState(false);
  const [regenerateDialogVisible, setRegenerateDialogVisible] = useState(false);
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  const [regenerateFormData, setRegenerateFormData] = useState<any>(null);
  const [regenerateForm] = Form.useForm();
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);

  const [knownTeamIDs, setKnownTeamIDs] = useState(initialKnownTeamIDs);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

  // Function to check if user is admin of a team
  const isUserTeamAdmin = (team: any) => {
    if (!team.members_with_roles) return false;
    return team.members_with_roles.some(
      (member: any) => member.role === "admin" && member.user_id === userID
    );
  };

  // Combine all keys that user should have access to
  const all_keys_to_display = React.useMemo(() => {
    let allKeys: any[] = [];

    // If no teams, return personal keys
    if (!teams || teams.length === 0) {
      return data;
    }

    teams.forEach((team) => {
      // For default team or when user is not admin, use personal keys (data)
      if (team.team_id === "default-team" || !isUserTeamAdmin(team)) {
        if (selectedTeam && selectedTeam.team_id === team.team_id && data) {
          allKeys = [
            ...allKeys,
            ...data.filter((key) => key.team_id === team.team_id),
          ];
        }
      }
      // For teams where user is admin, use team keys
      else if (isUserTeamAdmin(team)) {
        if (selectedTeam && selectedTeam.team_id === team.team_id) {
          allKeys = [
            ...allKeys,
            ...(data?.filter((key) => key?.team_id === team?.team_id) || []),
          ];
        }
      }
    });

    // If no team is selected, show all accessible keys
    if ((!selectedTeam || selectedTeam.team_alias === "Default Team") && data) {
      const personalKeys = data.filter(
        (key) => !key.team_id || key.team_id === "default-team"
      );
      const adminTeamKeys = teams
        .filter((team) => isUserTeamAdmin(team))
        .flatMap((team) => team.keys || []);
      allKeys = [...personalKeys, ...adminTeamKeys];
    }

    // Filter out litellm-dashboard keys
    allKeys = allKeys.filter((key) => key.team_id !== "litellm-dashboard");

    // Remove duplicates based on token
    const uniqueKeys = Array.from(
      new Map(allKeys.map((key) => [key.token, key])).values()
    );

    return uniqueKeys;
  }, [data, teams, selectedTeam, userID]);

  useEffect(() => {
    const calculateNewExpiryTime = (duration: string | undefined) => {
      if (!duration) {
        return null;
      }

      try {
        const now = new Date();
        let newExpiry: Date;

        if (duration.endsWith("s")) {
          newExpiry = add(now, { seconds: parseInt(duration) });
        } else if (duration.endsWith("h")) {
          newExpiry = add(now, { hours: parseInt(duration) });
        } else if (duration.endsWith("d")) {
          newExpiry = add(now, { days: parseInt(duration) });
        } else {
          throw new Error("Invalid duration format");
        }

        return newExpiry.toLocaleString("en-US", {
          year: "numeric",
          month: "numeric",
          day: "numeric",
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          hour12: true,
        });
      } catch (error) {
        return null;
      }
    };

    console.log("in calculateNewExpiryTime for selectedToken", selectedToken);

    // When a new duration is entered
    if (regenerateFormData?.duration) {
      setNewExpiryTime(calculateNewExpiryTime(regenerateFormData.duration));
    } else {
      setNewExpiryTime(null);
    }

    console.log("calculateNewExpiryTime:", newExpiryTime);
  }, [selectedToken, regenerateFormData?.duration]);

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }

        const models = await fetchAvailableModelsForTeamOrKey(
          userID,
          userRole,
          accessToken
        );
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const handleModelLimitClick = (token: ItemData) => {
    setSelectedToken(token);
    setModelLimitModalVisible(true);
  };

  const handleModelLimitSubmit = async (updatedMetadata: any) => {
    if (accessToken == null || selectedToken == null) {
      return;
    }

    const formValues = {
      ...selectedToken,
      metadata: updatedMetadata,
      key: selectedToken.token,
    };

    try {
      let newKeyValues = await keyUpdateCall(accessToken, formValues);
      console.log("Model limits updated:", newKeyValues);

      // Update the keys with the updated key
      if (data) {
        const updatedData = data.map((key) =>
          key.token === selectedToken.token ? newKeyValues : key
        );
        setData(updatedData);
      }
      message.success("Model-specific limits updated successfully");
    } catch (error) {
      console.error("Error updating model-specific limits:", error);
      message.error("Failed to update model-specific limits");
    }

    setModelLimitModalVisible(false);
    setSelectedToken(null);
  };

  useEffect(() => {
    if (teams) {
      const teamIDSet: Set<string> = new Set();
      teams.forEach((team: any, index: number) => {
        const team_obj: string = team.team_id;
        teamIDSet.add(team_obj);
      });
      setKnownTeamIDs(teamIDSet);
    }
  }, [teams]);
  const EditKeyModal: React.FC<EditKeyModalProps> = ({
    visible,
    onCancel,
    token,
    onSubmit,
  }) => {
    const [form] = Form.useForm();
    const [keyTeam, setKeyTeam] = useState(selectedTeam);
    const [errorModels, setErrorModels] = useState<string[]>([]);
    const [errorBudget, setErrorBudget] = useState<boolean>(false);
    const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

    useEffect(() => {
      const fetchGuardrails = async () => {
        try {
          const response = await getGuardrailsList(accessToken);
          const guardrailNames = response.guardrails.map(
            (g: { guardrail_name: string }) => g.guardrail_name
          );
          setGuardrailsList(guardrailNames);
        } catch (error) {
          console.error("Failed to fetch guardrails:", error);
        }
      };

      fetchGuardrails();
    }, [accessToken]);

    let metadataString = "";
    try {
      // Create a copy of metadata without guardrails for display
      const displayMetadata = { ...token.metadata };
      delete displayMetadata.guardrails;
      metadataString = JSON.stringify(displayMetadata, null, 2);
    } catch (error) {
      console.error("Error stringifying metadata:", error);
      metadataString = "";
    }

    // Extract existing guardrails from metadata
    let existingGuardrails: string[] = [];
    try {
      existingGuardrails = token.metadata?.guardrails || [];
    } catch (error) {
      console.error("Error extracting guardrails:", error);
    }

    const initialValues =
      token ?
        {
          ...token,
          budget_duration: token.budget_duration,
          metadata: metadataString,
          guardrails: existingGuardrails,
        }
      : { metadata: metadataString, guardrails: [] };

    const handleOk = () => {
      form
        .validateFields()
        .then((values) => {
          // const updatedValues = {...values, team_id: team.team_id};
          // onSubmit(updatedValues);
          form.resetFields();
        })
        .catch((error) => {
          console.error("Validation failed:", error);
        });
    };

    return (
      <Modal
        title="Edit Key"
        visible={visible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={onCancel}
      >
        <Form
          form={form}
          onFinish={handleEditSubmit}
          initialValues={initialValues}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item name="key_alias" label="Key Alias">
              <TextInput />
            </Form.Item>

            <Form.Item
              label="Models"
              name="models"
              rules={[
                {
                  validator: (rule, value) => {
                    if (keyTeam.team_alias === "Default Team") {
                      return Promise.resolve();
                    }

                    const errorModels = value.filter(
                      (model: string) =>
                        !keyTeam.models.includes(model) &&
                        model !== "all-team-models" &&
                        model !== "all-proxy-models" &&
                        !keyTeam.models.includes("all-proxy-models")
                    );
                    console.log(`errorModels: ${errorModels}`);
                    if (errorModels.length > 0) {
                      return Promise.reject(
                        `Some models are not part of the new team's models - ${errorModels} Team models: ${keyTeam.models}`
                      );
                    } else {
                      return Promise.resolve();
                    }
                  },
                },
              ]}
            >
              <Select
                mode="multiple"
                placeholder="Select models"
                style={{ width: "100%" }}
              >
                <Option key="all-team-models" value="all-team-models">
                  All Team Models
                </Option>
                {keyTeam.team_alias === "Default Team" ?
                  userModels
                    .filter((model) => model !== "all-proxy-models")
                    .map((model: string) => (
                      <Option key={model} value={model}>
                        {getModelDisplayName(model)}
                      </Option>
                    ))
                : keyTeam.models.map((model: string) => (
                    <Option key={model} value={model}>
                      {getModelDisplayName(model)}
                    </Option>
                  ))
                }
              </Select>
            </Form.Item>
            <Form.Item
              className="mt-8"
              label="Max Budget (USD)"
              name="max_budget"
              help={`Budget cannot exceed team max budget: ${keyTeam?.max_budget !== null && keyTeam?.max_budget !== undefined ? keyTeam?.max_budget : "unlimited"}`}
              rules={[
                {
                  validator: async (_, value) => {
                    if (
                      value &&
                      keyTeam &&
                      keyTeam.max_budget !== null &&
                      value > keyTeam.max_budget
                    ) {
                      console.log(`keyTeam.max_budget: ${keyTeam.max_budget}`);
                      throw new Error(
                        `Budget cannot exceed team max budget: $${keyTeam.max_budget}`
                      );
                    }
                  },
                },
              ]}
            >
              <InputNumber step={0.01} precision={2} width={200} />
            </Form.Item>

            <Form.Item
              className="mt-8"
              label="Reset Budget"
              name="budget_duration"
              help={`Current Reset Budget: ${
                token.budget_duration
              }, budget will be reset: ${token.budget_reset_at ? new Date(token.budget_reset_at).toLocaleString() : "Never"}`}
            >
              <Select placeholder="n/a">
                <Select.Option value="daily">daily</Select.Option>
                <Select.Option value="weekly">weekly</Select.Option>
                <Select.Option value="monthly">monthly</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item label="token" name="token" hidden={true}></Form.Item>
            <Form.Item
              label="Team"
              name="team_id"
              className="mt-8"
              help="the team this key belongs to"
            >
              <Select3 value={token.team_alias}>
                {teams?.map((team_obj, index) => (
                  <SelectItem
                    key={index}
                    value={team_obj.team_id}
                    onClick={() => setKeyTeam(team_obj)}
                  >
                    {team_obj.team_alias}
                  </SelectItem>
                ))}
              </Select3>
            </Form.Item>

            <Form.Item
              className="mt-8"
              label="TPM Limit (tokens per minute)"
              name="tpm_limit"
              help={`tpm_limit cannot exceed team tpm_limit ${keyTeam?.tpm_limit !== null && keyTeam?.tpm_limit !== undefined ? keyTeam?.tpm_limit : "unlimited"}`}
              rules={[
                {
                  validator: async (_, value) => {
                    if (
                      value &&
                      keyTeam &&
                      keyTeam.tpm_limit !== null &&
                      value > keyTeam.tpm_limit
                    ) {
                      console.log(`keyTeam.tpm_limit: ${keyTeam.tpm_limit}`);
                      throw new Error(
                        `tpm_limit cannot exceed team max tpm_limit: $${keyTeam.tpm_limit}`
                      );
                    }
                  },
                },
              ]}
            >
              <InputNumber step={1} precision={1} width={200} />
            </Form.Item>
            <Form.Item
              className="mt-8"
              label="RPM Limit (requests per minute)"
              name="rpm_limit"
              help={`rpm_limit cannot exceed team max rpm_limit: ${keyTeam?.rpm_limit !== null && keyTeam?.rpm_limit !== undefined ? keyTeam?.rpm_limit : "unlimited"}`}
              rules={[
                {
                  validator: async (_, value) => {
                    if (
                      value &&
                      keyTeam &&
                      keyTeam.rpm_limit !== null &&
                      value > keyTeam.rpm_limit
                    ) {
                      console.log(`keyTeam.rpm_limit: ${keyTeam.rpm_limit}`);
                      throw new Error(
                        `rpm_limit cannot exceed team max rpm_limit: $${keyTeam.rpm_limit}`
                      );
                    }
                  },
                },
              ]}
            >
              <InputNumber step={1} precision={1} width={200} />
            </Form.Item>
            <Form.Item
              label={
                <span>
                  Guardrails{" "}
                  <Tooltip title="Setup your first guardrail">
                    <a
                      href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </a>
                  </Tooltip>
                </span>
              }
              name="guardrails"
              className="mt-8"
              help="Select existing guardrails or enter new ones"
            >
              <Select
                mode="tags"
                style={{ width: "100%" }}
                placeholder="Select or enter guardrails"
                options={guardrailsList.map((name) => ({
                  value: name,
                  label: name,
                }))}
              />
            </Form.Item>
            <Form.Item
              label="Metadata (ensure this is valid JSON)"
              name="metadata"
            >
              <TextArea
                rows={10}
                onChange={(e) => {
                  form.setFieldsValue({ metadata: e.target.value });
                }}
              />
            </Form.Item>
          </>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Edit Key</Button2>
          </div>
        </Form>
      </Modal>
    );
  };

  const ModelLimitModal: React.FC<ModelLimitModalProps> = ({
    visible,
    onCancel,
    token,
    onSubmit,
    accessToken,
  }) => {
    const [modelLimits, setModelLimits] = useState<{
      [key: string]: { tpm: number; rpm: number };
    }>({});
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [newModelRow, setNewModelRow] = useState<string | null>(null);

    useEffect(() => {
      if (token.metadata) {
        const tpmLimits = token.metadata.model_tpm_limit || {};
        const rpmLimits = token.metadata.model_rpm_limit || {};
        const combinedLimits: { [key: string]: { tpm: number; rpm: number } } =
          {};

        Object.keys({ ...tpmLimits, ...rpmLimits }).forEach((model) => {
          combinedLimits[model] = {
            tpm: tpmLimits[model] || 0,
            rpm: rpmLimits[model] || 0,
          };
        });

        setModelLimits(combinedLimits);
      }

      const fetchAvailableModels = async () => {
        try {
          const modelDataResponse = await modelInfoCall(accessToken, "", "");
          const allModelGroups: string[] = Array.from(
            new Set(
              modelDataResponse.data.map((model: any) => model.model_name)
            )
          );
          setAvailableModels(allModelGroups);
        } catch (error) {
          console.error("Error fetching model data:", error);
          message.error("Failed to fetch available models");
        }
      };

      fetchAvailableModels();
    }, [token, accessToken]);

    const handleLimitChange = (
      model: string,
      type: "tpm" | "rpm",
      value: number | null
    ) => {
      setModelLimits((prev) => ({
        ...prev,
        [model]: {
          ...prev[model],
          [type]: value || 0,
        },
      }));
    };

    const handleAddLimit = () => {
      setNewModelRow("");
    };

    const handleModelSelect = (model: string) => {
      if (!modelLimits[model]) {
        setModelLimits((prev) => ({
          ...prev,
          [model]: { tpm: 0, rpm: 0 },
        }));
      }
      setNewModelRow(null);
    };

    const handleRemoveModel = (model: string) => {
      setModelLimits((prev) => {
        const { [model]: _, ...rest } = prev;
        return rest;
      });
    };

    const handleSubmit = () => {
      const updatedMetadata = {
        ...token.metadata,
        model_tpm_limit: Object.fromEntries(
          Object.entries(modelLimits).map(([model, limits]) => [
            model,
            limits.tpm,
          ])
        ),
        model_rpm_limit: Object.fromEntries(
          Object.entries(modelLimits).map(([model, limits]) => [
            model,
            limits.rpm,
          ])
        ),
      };
      onSubmit(updatedMetadata);
    };

    return (
      <Modal
        title="Edit Model-Specific Limits"
        visible={visible}
        onCancel={onCancel}
        footer={null}
        width={800}
      >
        <div className="space-y-4">
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Model</TableHeaderCell>
                <TableHeaderCell>TPM Limit</TableHeaderCell>
                <TableHeaderCell>RPM Limit</TableHeaderCell>
                <TableHeaderCell>Actions</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(modelLimits).map(([model, limits]) => (
                <TableRow key={model}>
                  <TableCell>{model}</TableCell>
                  <TableCell>
                    <InputNumber
                      value={limits.tpm}
                      onChange={(value) =>
                        handleLimitChange(model, "tpm", value)
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <InputNumber
                      value={limits.rpm}
                      onChange={(value) =>
                        handleLimitChange(model, "rpm", value)
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <Button onClick={() => handleRemoveModel(model)}>
                      Remove
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {newModelRow !== null && (
                <TableRow>
                  <TableCell>
                    <Select
                      style={{ width: 200 }}
                      placeholder="Select a model"
                      onChange={handleModelSelect}
                      value={newModelRow || undefined}
                    >
                      {availableModels
                        .filter((m) => !modelLimits.hasOwnProperty(m))
                        .map((m) => (
                          <Option key={m} value={m}>
                            {m}
                          </Option>
                        ))}
                    </Select>
                  </TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>
                    <Button onClick={() => setNewModelRow(null)}>Cancel</Button>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          <Button onClick={handleAddLimit} disabled={newModelRow !== null}>
            Add Limit
          </Button>
        </div>
        <div className="flex justify-end space-x-4 mt-6">
          <Button onClick={onCancel}>Cancel</Button>
          <Button onClick={handleSubmit}>Save</Button>
        </div>
      </Modal>
    );
  };

  const handleEditClick = (token: any) => {
    console.log("handleEditClick:", token);

    // set token.token to token.token_id if token_id is not null
    if (token.token == null) {
      if (token.token_id !== null) {
        token.token = token.token_id;
      }
    }

    // Convert the budget_duration to the corresponding select option
    let budgetDuration = null;
    if (token.budget_duration) {
      switch (token.budget_duration) {
        case "24h":
          budgetDuration = "daily";
          break;
        case "7d":
          budgetDuration = "weekly";
          break;
        case "30d":
          budgetDuration = "monthly";
          break;
        default:
          budgetDuration = "None";
      }
    }

    setSelectedToken({
      ...token,
      budget_duration: budgetDuration,
    });

    //setSelectedToken(token);
    setEditModalVisible(true);
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setSelectedToken(null);
  };

  const handleEditSubmit = async (formValues: Record<string, any>) => {
    /**
     * Call API to update team with teamId and values
     *
     * Client-side validation: For selected team, ensure models in team + max budget < team max budget
     */
    if (accessToken == null) {
      return;
    }

    const currentKey = formValues.token;
    formValues.key = currentKey;

    // Convert metadata back to an object if it exists and is a string
    if (formValues.metadata && typeof formValues.metadata === "string") {
      try {
        const parsedMetadata = JSON.parse(formValues.metadata);
        // Only add guardrails if they are set in form values
        formValues.metadata = {
          ...parsedMetadata,
          ...(formValues.guardrails?.length > 0 ?
            { guardrails: formValues.guardrails }
          : {}),
        };
      } catch (error) {
        console.error("Error parsing metadata JSON:", error);
        message.error(
          "Invalid metadata JSON for formValue " + formValues.metadata
        );
        return;
      }
    } else {
      // If metadata is not a string (or doesn't exist), only add guardrails if they are set
      formValues.metadata = {
        ...(formValues.metadata || {}),
        ...(formValues.guardrails?.length > 0 ?
          { guardrails: formValues.guardrails }
        : {}),
      };
    }

    // Convert the budget_duration back to the API expected format
    if (formValues.budget_duration) {
      switch (formValues.budget_duration) {
        case "daily":
          formValues.budget_duration = "24h";
          break;
        case "weekly":
          formValues.budget_duration = "7d";
          break;
        case "monthly":
          formValues.budget_duration = "30d";
          break;
      }
    }

    console.log("handleEditSubmit:", formValues);

    try {
      let newKeyValues = await keyUpdateCall(accessToken, formValues);
      console.log("handleEditSubmit: newKeyValues", newKeyValues);

      // Update the keys with the update key
      if (data) {
        const updatedData = data.map((key) =>
          key.token === currentKey ? newKeyValues : key
        );
        setData(updatedData);
      }
      message.success("Key updated successfully");

      setEditModalVisible(false);
      setSelectedToken(null);
    } catch (error) {
      console.error("Error updating key:", error);
      message.error("Failed to update key");
    }
  };

  const handleDelete = async (token: any) => {
    console.log("handleDelete:", token);
    if (token.token == null) {
      if (token.token_id !== null) {
        token.token = token.token_id;
      }
    }
    if (data == null) {
      return;
    }

    // Set the key to delete and open the confirmation modal
    setKeyToDelete(token.token);
    localStorage.removeItem("userData" + userID);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (keyToDelete == null || data == null) {
      return;
    }

    try {
      await keyDeleteCall(accessToken, keyToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      const filteredData = data.filter((item) => item.token !== keyToDelete);
      setData(filteredData);
    } catch (error) {
      console.error("Error deleting the key:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }

    // Close the confirmation modal and reset the keyToDelete
    setIsDeleteModalOpen(false);
    setKeyToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the keyToDelete
    setIsDeleteModalOpen(false);
    setKeyToDelete(null);
  };

  const handleRegenerateClick = (token: any) => {
    setSelectedToken(token);
    setNewExpiryTime(null);
    regenerateForm.setFieldsValue({
      key_alias: token.key_alias,
      max_budget: token.max_budget,
      tpm_limit: token.tpm_limit,
      rpm_limit: token.rpm_limit,
      duration: token.duration || "",
    });
    setRegenerateDialogVisible(true);
  };

  const handleRegenerateFormChange = (field: string, value: any) => {
    setRegenerateFormData((prev: any) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleRegenerateKey = async () => {
    if (!premiumUser) {
      message.error(
        "Regenerate API Key is an Enterprise feature. Please upgrade to use this feature."
      );
      return;
    }

    if (selectedToken == null) {
      return;
    }

    try {
      const formValues = await regenerateForm.validateFields();
      const response = await regenerateKeyCall(
        accessToken,
        selectedToken.token,
        formValues
      );
      setRegeneratedKey(response.key);

      // Update the data state with the new key_name
      if (data) {
        const updatedData = data.map((item) =>
          item.token === selectedToken?.token ?
            { ...item, key_name: response.key_name, ...formValues }
          : item
        );
        setData(updatedData);
      }

      setRegenerateDialogVisible(false);
      regenerateForm.resetFields();
      message.success("API Key regenerated successfully");
    } catch (error) {
      console.error("Error regenerating key:", error);
      message.error("Failed to regenerate API Key");
    }
  };

  if (data == null) {
    return;
  }
  console.log("RERENDER TRIGGERED");
  return (
    <div>
      <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh] mb-4 mt-2">
        <Table className="mt-5 max-h-[300px] min-h-[300px]">
          <TableHead>
            <TableRow>
              <TableHeaderCell>Key Alias</TableHeaderCell>
              <TableHeaderCell>Secret Key</TableHeaderCell>
              <TableHeaderCell>Created</TableHeaderCell>
              <TableHeaderCell>Expires</TableHeaderCell>
              <TableHeaderCell>Spend (USD)</TableHeaderCell>
              <TableHeaderCell>Budget (USD)</TableHeaderCell>
              <TableHeaderCell>Budget Reset</TableHeaderCell>
              <TableHeaderCell>Models</TableHeaderCell>
              <TableHeaderCell>Rate Limits</TableHeaderCell>
              <TableHeaderCell>Rate Limits per model</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {all_keys_to_display &&
              all_keys_to_display.map((item) => {
                console.log(item);
                // skip item if item.team_id == "litellm-dashboard"
                if (item.team_id === "litellm-dashboard") {
                  return null;
                }
                if (selectedTeam) {
                  /**
                   * if selected team id is null -> show the keys with no team id or team id's that don't exist in db
                   */
                  console.log(
                    `item team id: ${item.team_id}, knownTeamIDs.has(item.team_id): ${knownTeamIDs.has(item.team_id)}, selectedTeam id: ${selectedTeam.team_id}`
                  );
                  if (
                    selectedTeam.team_id == null &&
                    item.team_id !== null &&
                    !knownTeamIDs.has(item.team_id)
                  ) {
                    // do nothing -> returns a row with this key
                  } else if (item.team_id != selectedTeam.team_id) {
                    return null;
                  }
                  console.log(`item team id: ${item.team_id}, is returned`);
                }
                return (
                  <TableRow key={item.token}>
                    <TableCell
                      style={{
                        maxWidth: "2px",
                        whiteSpace: "pre-wrap",
                        overflow: "hidden",
                      }}
                    >
                      {item.key_alias != null ?
                        <Text>{item.key_alias}</Text>
                      : <Text>Not Set</Text>}
                    </TableCell>
                    <TableCell>
                      <Text>{item.key_name}</Text>
                    </TableCell>
                    <TableCell>
                      {item.created_at != null ?
                        <div>
                          <p style={{ fontSize: "0.70rem" }}>
                            {new Date(item.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      : <p style={{ fontSize: "0.70rem" }}>Not available</p>}
                    </TableCell>
                    <TableCell>
                      {item.expires != null ?
                        <div>
                          <p style={{ fontSize: "0.70rem" }}>
                            {new Date(item.expires).toLocaleDateString()}
                          </p>
                        </div>
                      : <p style={{ fontSize: "0.70rem" }}>Never</p>}
                    </TableCell>
                    <TableCell>
                      <Text>
                        {(() => {
                          try {
                            return parseFloat(item.spend).toFixed(4);
                          } catch (error) {
                            return item.spend;
                          }
                        })()}
                      </Text>
                    </TableCell>
                    <TableCell>
                      {item.max_budget != null ?
                        <Text>{item.max_budget}</Text>
                      : <Text>Unlimited</Text>}
                    </TableCell>
                    <TableCell>
                      {item.budget_reset_at != null ?
                        <div>
                          <p style={{ fontSize: "0.70rem" }}>
                            {new Date(item.budget_reset_at).toLocaleString()}
                          </p>
                        </div>
                      : <p style={{ fontSize: "0.70rem" }}>Never</p>}
                    </TableCell>
                    {/* <TableCell style={{ maxWidth: '2px' }}>
                  <ViewKeySpendReport
                    token={item.token}
                    accessToken={accessToken}
                    keySpend={item.spend}
                    keyBudget={item.max_budget}
                    keyName={item.key_name}
                  />
                </TableCell> */}
                    {/* <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>
                  <Text>{item.team_alias && item.team_alias != "None" ? item.team_alias : item.team_id}</Text>
                </TableCell> */}
                    {/* <TableCell style={{ maxWidth: "4px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>
                  <Text>{JSON.stringify(item.metadata).slice(0, 400)}</Text>
                  
                </TableCell> */}

                    <TableCell>
                      {Array.isArray(item.models) ?
                        <div
                          style={{ display: "flex", flexDirection: "column" }}
                        >
                          {item.models.length === 0 ?
                            <>
                              {
                                (
                                  selectedTeam &&
                                  selectedTeam.models &&
                                  selectedTeam.models.length > 0
                                ) ?
                                  selectedTeam.models.map(
                                    (model: string, index: number) =>
                                      model === "all-proxy-models" ?
                                        <Badge
                                          key={index}
                                          size={"xs"}
                                          className="mb-1"
                                          color="red"
                                        >
                                          <Text>All Proxy Models</Text>
                                        </Badge>
                                      : model === "all-team-models" ?
                                        <Badge
                                          key={index}
                                          size={"xs"}
                                          className="mb-1"
                                          color="red"
                                        >
                                          <Text>All Team Models</Text>
                                        </Badge>
                                      : <Badge
                                          key={index}
                                          size={"xs"}
                                          className="mb-1"
                                          color="blue"
                                        >
                                          <Text>
                                            {model.length > 30 ?
                                              `${getModelDisplayName(model).slice(0, 30)}...`
                                            : getModelDisplayName(model)}
                                          </Text>
                                        </Badge>
                                  )
                                  // If selected team is None or selected team's models are empty, show all models
                                : <Badge
                                    size={"xs"}
                                    className="mb-1"
                                    color="blue"
                                  >
                                    <Text>all-proxy-models</Text>
                                  </Badge>

                              }
                            </>
                          : item.models.map((model: string, index: number) =>
                              model === "all-proxy-models" ?
                                <Badge
                                  key={index}
                                  size={"xs"}
                                  className="mb-1"
                                  color="red"
                                >
                                  <Text>All Proxy Models</Text>
                                </Badge>
                              : model === "all-team-models" ?
                                <Badge
                                  key={index}
                                  size={"xs"}
                                  className="mb-1"
                                  color="red"
                                >
                                  <Text>All Team Models</Text>
                                </Badge>
                              : <Badge
                                  key={index}
                                  size={"xs"}
                                  className="mb-1"
                                  color="blue"
                                >
                                  <Text>
                                    {model.length > 30 ?
                                      `${getModelDisplayName(model).slice(0, 30)}...`
                                    : getModelDisplayName(model)}
                                  </Text>
                                </Badge>
                            )
                          }
                        </div>
                      : null}
                    </TableCell>

                    <TableCell>
                      <Text>
                        TPM: {item.tpm_limit ? item.tpm_limit : "Unlimited"}{" "}
                        <br></br> RPM:{" "}
                        {item.rpm_limit ? item.rpm_limit : "Unlimited"}
                      </Text>
                    </TableCell>
                    <TableCell>
                      <Button
                        size="xs"
                        onClick={() => handleModelLimitClick(item)}
                      >
                        Edit Limits
                      </Button>
                    </TableCell>
                    <TableCell>
                      <Icon
                        onClick={() => {
                          setSelectedToken(item);
                          setInfoDialogVisible(true);
                        }}
                        icon={InformationCircleIcon}
                        size="sm"
                      />

                      <Modal
                        open={infoDialogVisible}
                        onCancel={() => {
                          setInfoDialogVisible(false);
                          setSelectedToken(null);
                        }}
                        footer={null}
                        width={800}
                      >
                        {selectedToken && (
                          <>
                            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 mt-8">
                              <Card>
                                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                                  Spend
                                </p>
                                <div className="mt-2 flex items-baseline space-x-2.5">
                                  <p className="text-tremor font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                                    {(() => {
                                      try {
                                        return parseFloat(
                                          selectedToken.spend
                                        ).toFixed(4);
                                      } catch (error) {
                                        return selectedToken.spend;
                                      }
                                    })()}
                                  </p>
                                </div>
                              </Card>
                              <Card key={item.name}>
                                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                                  Budget
                                </p>
                                <div className="mt-2 flex items-baseline space-x-2.5">
                                  <p className="text-tremor font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                                    {selectedToken.max_budget != null ?
                                      <>
                                        {selectedToken.max_budget}
                                        {selectedToken.budget_duration && (
                                          <>
                                            <br />
                                            Budget will be reset at{" "}
                                            {selectedToken.budget_reset_at ?
                                              new Date(
                                                selectedToken.budget_reset_at
                                              ).toLocaleString()
                                            : "Never"}
                                          </>
                                        )}
                                      </>
                                    : <>Unlimited</>}
                                  </p>
                                </div>
                              </Card>
                              <Card key={item.name}>
                                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                                  Expires
                                </p>
                                <div className="mt-2 flex items-baseline space-x-2.5">
                                  <p className="text-tremor-default font-small text-tremor-content-strong dark:text-dark-tremor-content-strong">
                                    {selectedToken.expires != null ?
                                      <>
                                        {new Date(
                                          selectedToken.expires
                                        ).toLocaleString(undefined, {
                                          day: "numeric",
                                          month: "long",
                                          year: "numeric",
                                          hour: "numeric",
                                          minute: "numeric",
                                          second: "numeric",
                                        })}
                                      </>
                                    : <>Never</>}
                                  </p>
                                </div>
                              </Card>
                            </div>

                            <Card className="my-4">
                              <Title>Token Name</Title>
                              <Text className="my-1">
                                {selectedToken.key_alias ?
                                  selectedToken.key_alias
                                : selectedToken.key_name}
                              </Text>
                              <Title>Token ID</Title>
                              <Text className="my-1 text-[12px]">
                                {selectedToken.token}
                              </Text>
                              <Title>User ID</Title>
                              <Text className="my-1 text-[12px]">
                                {selectedToken.user_id}
                              </Text>
                              <Title>Metadata</Title>
                              <Text className="my-1">
                                <pre>
                                  {JSON.stringify(selectedToken.metadata)}{" "}
                                </pre>
                              </Text>
                            </Card>

                            <Button
                              className="mx-auto flex items-center"
                              onClick={() => {
                                setInfoDialogVisible(false);
                                setSelectedToken(null);
                              }}
                            >
                              Close
                            </Button>
                          </>
                        )}
                      </Modal>
                      <Icon
                        icon={PencilAltIcon}
                        size="sm"
                        onClick={() => handleEditClick(item)}
                      />
                      <Icon
                        onClick={() => handleRegenerateClick(item)}
                        icon={RefreshIcon}
                        size="sm"
                      />
                      <Icon
                        onClick={() => handleDelete(item)}
                        icon={TrashIcon}
                        size="sm"
                      />
                    </TableCell>
                  </TableRow>
                );
              })}
          </TableBody>
        </Table>
        {isDeleteModalOpen && (
          <div className="fixed z-10 inset-0 overflow-y-auto">
            <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
              <div
                className="fixed inset-0 transition-opacity"
                aria-hidden="true"
              >
                <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
              </div>

              {/* Modal Panel */}
              <span
                className="hidden sm:inline-block sm:align-middle sm:h-screen"
                aria-hidden="true"
              >
                &#8203;
              </span>

              {/* Confirmation Modal Content */}
              <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                  <div className="sm:flex sm:items-start">
                    <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                      <h3 className="text-lg leading-6 font-medium text-gray-900">
                        Delete Key
                      </h3>
                      <div className="mt-2">
                        <p className="text-sm text-gray-500">
                          Are you sure you want to delete this key ?
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                  <Button onClick={confirmDelete} color="red" className="ml-2">
                    Delete
                  </Button>
                  <Button onClick={cancelDelete}>Cancel</Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>

      {selectedToken && (
        <EditKeyModal
          visible={editModalVisible}
          onCancel={handleEditCancel}
          token={selectedToken}
          onSubmit={handleEditSubmit}
        />
      )}

      {selectedToken && (
        <ModelLimitModal
          visible={modelLimitModalVisible}
          onCancel={() => setModelLimitModalVisible(false)}
          token={selectedToken}
          onSubmit={handleModelLimitSubmit}
          accessToken={accessToken}
        />
      )}

      {/* Regenerate Key Form Modal */}
      <Modal
        title="Regenerate API Key"
        visible={regenerateDialogVisible}
        onCancel={() => {
          setRegenerateDialogVisible(false);
          regenerateForm.resetFields();
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setRegenerateDialogVisible(false);
              regenerateForm.resetFields();
            }}
            className="mr-2"
          >
            Cancel
          </Button>,
          <Button
            key="regenerate"
            onClick={handleRegenerateKey}
            disabled={!premiumUser}
          >
            {premiumUser ? "Regenerate" : "Upgrade to Regenerate"}
          </Button>,
        ]}
      >
        {premiumUser ?
          <Form
            form={regenerateForm}
            layout="vertical"
            onValuesChange={(changedValues, allValues) => {
              if ("duration" in changedValues) {
                handleRegenerateFormChange("duration", changedValues.duration);
              }
            }}
          >
            <Form.Item name="key_alias" label="Key Alias">
              <TextInput disabled={true} />
            </Form.Item>
            <Form.Item name="max_budget" label="Max Budget (USD)">
              <InputNumber
                step={0.01}
                precision={2}
                style={{ width: "100%" }}
              />
            </Form.Item>
            <Form.Item name="tpm_limit" label="TPM Limit">
              <InputNumber style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="rpm_limit" label="RPM Limit">
              <InputNumber style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item
              name="duration"
              label="Expire Key (eg: 30s, 30h, 30d)"
              className="mt-8"
            >
              <TextInput placeholder="" />
            </Form.Item>
            <div className="mt-2 text-sm text-gray-500">
              Current expiry:{" "}
              {selectedToken?.expires != null ?
                new Date(selectedToken.expires).toLocaleString()
              : "Never"}
            </div>
            {newExpiryTime && (
              <div className="mt-2 text-sm text-green-600">
                New expiry: {newExpiryTime}
              </div>
            )}
          </Form>
        : <div>
            <p className="mb-2 text-gray-500 italic text-[12px]">
              Upgrade to use this feature
            </p>
            <Button variant="primary" className="mb-2">
              <a
                href="https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
                target="_blank"
              >
                Get Free Trial
              </a>
            </Button>
          </div>
        }
      </Modal>

      {/* Regenerated Key Display Modal */}
      {regeneratedKey && (
        <Modal
          visible={!!regeneratedKey}
          onCancel={() => setRegeneratedKey(null)}
          footer={[
            <Button key="close" onClick={() => setRegeneratedKey(null)}>
              Close
            </Button>,
          ]}
        >
          <Grid numItems={1} className="gap-2 w-full">
            <Title>Regenerated Key</Title>
            <Col numColSpan={1}>
              <p>
                Please replace your old key with the new key generated. For
                security reasons, <b>you will not be able to view it again</b>{" "}
                through your LiteLLM account. If you lose this secret key, you
                will need to generate a new one.
              </p>
            </Col>
            <Col numColSpan={1}>
              <Text className="mt-3">Key Alias:</Text>
              <div
                style={{
                  background: "#f8f8f8",
                  padding: "10px",
                  borderRadius: "5px",
                  marginBottom: "10px",
                }}
              >
                <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>
                  {selectedToken?.key_alias || "No alias set"}
                </pre>
              </div>
              <Text className="mt-3">New API Key:</Text>
              <div
                style={{
                  background: "#f8f8f8",
                  padding: "10px",
                  borderRadius: "5px",
                  marginBottom: "10px",
                }}
              >
                <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>
                  {regeneratedKey}
                </pre>
              </div>
              <CopyToClipboard
                text={regeneratedKey}
                onCopy={() => message.success("API Key copied to clipboard")}
              >
                <Button className="mt-3">Copy API Key</Button>
              </CopyToClipboard>
            </Col>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

export default ViewKeyTable;
