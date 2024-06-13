"use client";
import React, { useEffect, useState } from "react";
import { keyDeleteCall, modelAvailableCall } from "./networking";
import { InformationCircleIcon, StatusOnlineIcon, TrashIcon, PencilAltIcon } from "@heroicons/react/outline";
import { keySpendLogsCall, PredictedSpendLogsCall, keyUpdateCall } from "./networking";
import {
  Badge,
  Card,
  Table,
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
} from "@tremor/react";
import { Select as Select3, SelectItem, MultiSelect, MultiSelectItem } from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
  Select,
} from "antd";

const { Option } = Select;


interface EditKeyModalProps {
  visible: boolean;
  onCancel: () => void;
  token: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
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
  expires: any;
  // Add any other properties that exist in the item data
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  userRole,
  accessToken,
  selectedTeam,
  data,
  setData,
  teams
}) => {
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<ItemData | null>(null);
  const [spendData, setSpendData] = useState<{ day: string; spend: number }[] | null>(
    null
  );
  const [predictedSpendString, setPredictedSpendString] = useState("");

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [infoDialogVisible, setInfoDialogVisible] = useState(false);
  const [selectedToken, setSelectedToken] = useState<ItemData | null>(null);
  const [userModels, setUserModels] = useState([]);
  const initialKnownTeamIDs: Set<string> = new Set();

  const [knownTeamIDs, setKnownTeamIDs] = useState(initialKnownTeamIDs);

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null) {
          return;
        }

        if (accessToken !== null && userRole !== null) {
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          let available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
          console.log("available_model_names:", available_model_names);
          setUserModels(available_model_names);
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
        const team_obj: string = team.team_id
        teamIDSet.add(team_obj);
      });
      setKnownTeamIDs(teamIDSet)
    }
  }, [teams])
  const EditKeyModal: React.FC<EditKeyModalProps> = ({ visible, onCancel, token, onSubmit }) => {
    const [form] = Form.useForm();
    const [keyTeam, setKeyTeam] = useState(selectedTeam);
    const [errorModels, setErrorModels] = useState<string[]>([]);
    const [errorBudget, setErrorBudget] = useState<boolean>(false);

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
          initialValues={token} // Pass initial values here
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
                <>
                <Form.Item 
                label="Key Name" 
                name="key_alias"
                rules={[{ required: true, message: 'Please input a key name' }]}
                help="required"
              >
                <Input />
              </Form.Item>

              <Form.Item label="Models" name="models" rules={[
                {
                  validator: (rule, value) => {
                    const errorModels = value.filter((model: string) => (
                      !keyTeam.models.includes(model) && 
                      model !== "all-team-models" && 
                      model !== "all-proxy-models" && 
                      !keyTeam.models.includes("all-proxy-models")
                    ));
                    console.log(`errorModels: ${errorModels}`)
                    if (errorModels.length > 0) {
                      return Promise.reject(`Some models are not part of the new team\'s models - ${errorModels}Team models: ${keyTeam.models}`);
                    } else {
                      return Promise.resolve();
                    }
                  }
                }
              ]}>
                <Select
                  mode="multiple"
                  placeholder="Select models"
                  style={{ width: "100%" }}
                >
                  <Option key="all-team-models" value="all-team-models">
                    All Team Models
                  </Option>                
                  {keyTeam && keyTeam.models ? (
                    keyTeam.models.includes("all-proxy-models") ? (
                      userModels.filter(model => model !== "all-proxy-models").map((model: string) => (
                        <Option key={model} value={model}>
                          {model}
                        </Option>
                      ))
                    ) : (
                      keyTeam.models.map((model: string) => (
                        <Option key={model} value={model}>
                          {model}
                        </Option>
                      ))
                    )
                  ) : (
                    userModels.map((model: string) => (
                      <Option key={model} value={model}>
                        {model}
                      </Option>
                    ))
                  )}
                </Select>
              </Form.Item>
              <Form.Item 
                className="mt-8"
                label="Max Budget (USD)" 
                name="max_budget" 
                help={`Budget cannot exceed team max budget: ${keyTeam?.max_budget !== null && keyTeam?.max_budget !== undefined ? keyTeam?.max_budget : 'unlimited'}`}
                rules={[
                  {
                      validator: async (_, value) => {
                          if (value && keyTeam && keyTeam.max_budget !== null && value > keyTeam.max_budget) {
                              console.log(`keyTeam.max_budget: ${keyTeam.max_budget}`)
                              throw new Error(`Budget cannot exceed team max budget: $${keyTeam.max_budget}`);
                          }
                      },
                  },
              ]}
              >
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item
                  label="token"
                  name="token"
                  hidden={true}
                ></Form.Item>
              <Form.Item 
                label="Team" 
                name="team_id"
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
            </>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Edit Key</Button2>
          </div>
        </Form>
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

    setSelectedToken(token);
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

  console.log("handleEditSubmit:", formValues);

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
            <TableHeaderCell>Spend (USD)</TableHeaderCell>
            <TableHeaderCell>Budget (USD)</TableHeaderCell>
            <TableHeaderCell>Models</TableHeaderCell>
            <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((item) => {
            console.log(item);
            // skip item if item.team_id == "litellm-dashboard"
            if (item.team_id === "litellm-dashboard") {
              return null;
            }
            if (selectedTeam) {
              /**
               * if selected team id is null -> show the keys with no team id or team id's that don't exist in db
               */
              console.log(`item team id: ${item.team_id}, knownTeamIDs.has(item.team_id): ${knownTeamIDs.has(item.team_id)}, selectedTeam id: ${selectedTeam.team_id}`)
              if (selectedTeam.team_id == null && item.team_id !== null && !knownTeamIDs.has(item.team_id)) {
                // do nothing -> returns a row with this key
              }
              else if (item.team_id != selectedTeam.team_id) {
                return null;
              }
              console.log(`item team id: ${item.team_id}, is returned`)
            }
            return (
              <TableRow key={item.token}>
                <TableCell style={{ maxWidth: "2px", whiteSpace: "pre-wrap", overflow: "hidden"  }}>
                  {item.key_alias != null ? (
                    <Text>{item.key_alias}</Text>
                  ) : (
                    <Text>Not Set</Text>
                  )}
                </TableCell>
                <TableCell>
                  <Text>{item.key_name}</Text>
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
                  {item.max_budget != null ? (
                    <Text>{item.max_budget}</Text>
                  ) : (
                    <Text>Unlimited</Text>
                  )}
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
  {Array.isArray(item.models) ? (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {item.models.length === 0 ? (
        <>
          {selectedTeam && selectedTeam.models && selectedTeam.models.length > 0 ? (
            selectedTeam.models.map((model: string, index: number) => (
              model === "all-proxy-models" ? (
                <Badge key={index} size={"xs"} className="mb-1" color="red">
                  <Text>All Proxy Models</Text>
                </Badge>
              ) : model === "all-team-models" ? (
                <Badge key={index} size={"xs"} className="mb-1" color="red">
                  <Text>All Team Models</Text>
                </Badge>
              ) : (
                <Badge key={index} size={"xs"} className="mb-1" color="blue">
                  <Text>{model.length > 30 ? `${model.slice(0, 30)}...` : model}</Text>
                </Badge>
              )
            ))
          ) : (
            // If selected team is None or selected team's models are empty, show all models
            <Badge size={"xs"} className="mb-1" color="blue">
              <Text>all-proxy-models</Text>
            </Badge>
          )}
        </>
      ) : (
        item.models.map((model: string, index: number) => (
          model === "all-proxy-models" ? (
            <Badge key={index} size={"xs"} className="mb-1" color="red">
              <Text>All Proxy Models</Text>
            </Badge>
          ) : model === "all-team-models" ? (
            <Badge key={index} size={"xs"} className="mb-1" color="red">
              <Text>All Team Models</Text>
            </Badge>
          ) : (
            <Badge key={index} size={"xs"} className="mb-1" color="blue">
              <Text>{model.length > 30 ? `${model.slice(0, 30)}...` : model}</Text>
            </Badge>
          )
        ))
      )}
    </div>
  ) : null}
</TableCell>

                <TableCell>
                  <Text>
                    TPM: {item.tpm_limit ? item.tpm_limit : "Unlimited"}{" "}
                    <br></br> RPM:{" "}
                    {item.rpm_limit ? item.rpm_limit : "Unlimited"}
                  </Text>
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
                        return parseFloat(selectedToken.spend).toFixed(4);
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
              {selectedToken.max_budget != null ? (
                  <>{selectedToken.max_budget}</>
                ) : (
                  <>Unlimited</>
                )}
              </p>
            </div>
          </Card>
          <Card key={item.name}>
            <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
              Expires
            </p>
            <div className="mt-2 flex items-baseline space-x-2.5">
              <p className="text-tremor-default font-small text-tremor-content-strong dark:text-dark-tremor-content-strong">
              {selectedToken.expires != null ? (
                  <>
                  {new Date(selectedToken.expires).toLocaleString(undefined, {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                    hour: 'numeric',
                    minute: 'numeric',
                    second: 'numeric'
                  })}
                </>
                ) : (
                  <>Never</>
                )}
              </p>
            </div>
          </Card>
      </div>

      <Card className="my-4">
        <Title>Token Name</Title>
        <Text className="my-1">{selectedToken.key_alias ? selectedToken.key_alias : selectedToken.key_name}</Text>
        <Title>Token ID</Title>
        <Text className="my-1 text-[12px]">{selectedToken.token}</Text>              
        <Title>Metadata</Title>
        <Text className="my-1"><pre>{JSON.stringify(selectedToken.metadata)} </pre></Text>
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
    </div>
  );
};

export default ViewKeyTable;
