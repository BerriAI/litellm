"use client";
import React, { useEffect, useState, useMemo } from "react";
import {
  keyDeleteCall,
  modelAvailableCall,
  getGuardrailsList,
  Organization,
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
  Select,
  SelectItem,
} from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import {
  fetchAvailableModelsForTeamOrKey,
  getModelDisplayName,
} from "./key_team_helpers/fetch_available_models_team_key";
import {
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
  Tooltip,
  DatePicker,
} from "antd";
import { CopyToClipboard } from "react-copy-to-clipboard";
import TextArea from "antd/es/input/TextArea";
import useKeyList from "./key_team_helpers/key_list";
import { KeyResponse } from "./key_team_helpers/key_list";
import { AllKeysTable } from "./all_keys_table";
import { Team } from "./key_team_helpers/key_list";
import { Setter } from "@/types";

interface EditKeyModalProps {
  visible: boolean;
  onCancel: () => void;
  token: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

interface ModelLimitModalProps {
  visible: boolean;
  onCancel: () => void;
  token: KeyResponse;
  onSubmit: (updatedMetadata: any) => void;
  accessToken: string;
}

// Define the props type
interface ViewKeyTableProps {
  userID: string;
  userRole: string | null;
  accessToken: string;
  selectedTeam: any | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
  teams: Team[] | null;
  premiumUser: boolean;
  currentOrg: Organization | null;
  organizations: Organization[] | null;
  setCurrentOrg: React.Dispatch<React.SetStateAction<Organization | null>>;
  selectedKeyAlias: string | null;
  setSelectedKeyAlias: Setter<string | null>;
  createClicked: boolean;
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

interface ModelLimits {
  [key: string]: number;  // Index signature allowing string keys
}

interface CombinedLimit {
  tpm: number;
  rpm: number;
}

interface CombinedLimits {
  [key: string]: CombinedLimit;  // Index signature allowing string keys
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  userRole,
  accessToken,
  selectedTeam,
  setSelectedTeam,
  data,
  setData,
  teams,
  premiumUser,
  currentOrg,
  organizations,
  setCurrentOrg,
  selectedKeyAlias,
  setSelectedKeyAlias,
  createClicked
}) => {
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<KeyResponse | null>(null);
  const [spendData, setSpendData] = useState<
    { day: string; spend: number }[] | null
  >(null);
  
  // NEW: Declare filter states for team and key alias.
  const [teamFilter, setTeamFilter] = useState<string>(selectedTeam?.team_id || "");


  // Keep the team filter in sync with the incoming prop.
  useEffect(() => {
    setTeamFilter(selectedTeam?.team_id || "");
  }, [selectedTeam]);

  // Build a memoized filters object for the backend call.

  // Pass filters into the hook so the API call includes these query parameters.
  const { keys, isLoading, error, pagination, refresh, setKeys } = useKeyList({
    selectedTeam,
    currentOrg,
    selectedKeyAlias,
    accessToken,
    createClicked,
  });


  const handlePageChange = (newPage: number) => {
    refresh({ page: newPage });
  };

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [infoDialogVisible, setInfoDialogVisible] = useState(false);
  const [selectedToken, setSelectedToken] = useState<KeyResponse | null>(null);
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


  return (
    <div>
      <AllKeysTable 
        keys={keys}
        setKeys={setKeys}
        isLoading={isLoading}
        pagination={pagination}
        onPageChange={handlePageChange}
        pageSize={100}
        teams={teams}
        selectedTeam={selectedTeam}
        setSelectedTeam={setSelectedTeam}
        accessToken={accessToken}
        userID={userID}
        userRole={userRole}
        organizations={organizations}
        setCurrentOrg={setCurrentOrg}
        refresh={refresh}
        selectedKeyAlias={selectedKeyAlias}
        setSelectedKeyAlias={setSelectedKeyAlias}
        premiumUser={premiumUser}
      />

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

// Update the type declaration to include the new function
declare global {
  interface Window {
    refreshKeysList?: () => void;
    addNewKeyToList?: (newKey: any) => void;
  }
}

export default ViewKeyTable;
