/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
import React, { useState, useEffect } from "react";
import { Typography } from "antd";
import { useRouter } from "next/navigation";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
} from "antd";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Select, SelectItem, Subtitle } from "@tremor/react";
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
  Col,
  Text,
  Grid,
  Callout,
  Divider,
} from "@tremor/react";
import { PencilAltIcon } from "@heroicons/react/outline";
import OnboardingModal from "./onboarding_link";
import { InvitationLink } from "./onboarding_link";
interface AdminPanelProps {
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
  showSSOBanner: boolean;
  premiumUser: boolean;
}

import {
  userUpdateUserCall,
  Member,
  userGetAllUsersCall,
  User,
  setCallbacksCall,
  invitationCreateCall,
  getPossibleUserRoles,
  addAllowedIP,
  getAllowedIPs,
  deleteAllowedIP,
} from "./networking";

const AdminPanel: React.FC<AdminPanelProps> = ({
  searchParams,
  accessToken,
  showSSOBanner,
  premiumUser,
}) => {
  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  const [admins, setAdmins] = useState<null | any[]>(null);
  const [invitationLinkData, setInvitationLinkData] =
    useState<InvitationLink | null>(null);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] =
    useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isAddAdminModalVisible, setIsAddAdminModalVisible] = useState(false);
  const [isUpdateMemberModalVisible, setIsUpdateModalModalVisible] =
    useState(false);
  const [isAddSSOModalVisible, setIsAddSSOModalVisible] = useState(false);
  const [isInstructionsModalVisible, setIsInstructionsModalVisible] =
    useState(false);
  const [isAllowedIPModalVisible, setIsAllowedIPModalVisible] = useState(false);
  const [isAddIPModalVisible, setIsAddIPModalVisible] = useState(false);
  const [isDeleteIPModalVisible, setIsDeleteIPModalVisible] = useState(false);
  const [allowedIPs, setAllowedIPs] = useState<string[]>([]);
  const [ipToDelete, setIPToDelete] = useState<string | null>(null);
  const router = useRouter();

  const [possibleUIRoles, setPossibleUIRoles] = useState<null | Record<
    string,
    Record<string, string>
  >>(null);

  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function() {};
  }
  const [baseUrl, setBaseUrl] = useState(
    isLocal ? "http://localhost:4000" : ""
  );

  const all_ip_address_allowed = "All IP Addresses Allowed";

  let nonSssoUrl;
  try {
    nonSssoUrl = window.location.origin;
  } catch (error) {
    nonSssoUrl = "<your-proxy-url>";
  }
  nonSssoUrl += "/fallback/login";

  const handleShowAllowedIPs = async () => {
    try {
      if (premiumUser !== true) {
        message.error(
          "This feature is only available for premium users. Please upgrade your account."
        )
        return
      }
      if (accessToken) {
        const data = await getAllowedIPs(accessToken);
        setAllowedIPs(data && data.length > 0 ? data : [all_ip_address_allowed]);
      } else {
        setAllowedIPs([all_ip_address_allowed]);
      }
    } catch (error) {
      console.error("Error fetching allowed IPs:", error);
      message.error(`Failed to fetch allowed IPs ${error}`);
      setAllowedIPs([all_ip_address_allowed]);
    } finally {
      if (premiumUser === true) {
        setIsAllowedIPModalVisible(true);
      }
    }
  };
  
  const handleAddIP = async (values: { ip: string }) => {
    try {
      if (accessToken) {
        await addAllowedIP(accessToken, values.ip);
        // Fetch the updated list of IPs
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs);
        message.success('IP address added successfully');
      }
    } catch (error) {
      console.error("Error adding IP:", error);
      message.error(`Failed to add IP address ${error}`);
    } finally {
      setIsAddIPModalVisible(false);
    }
  };
  
  const handleDeleteIP = async (ip: string) => {
    setIPToDelete(ip);
    setIsDeleteIPModalVisible(true);
  };
  
  const confirmDeleteIP = async () => {
    if (ipToDelete && accessToken) {
      try {
        await deleteAllowedIP(accessToken, ipToDelete);
        // Fetch the updated list of IPs
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs.length > 0 ? updatedIPs : [all_ip_address_allowed]);
        message.success('IP address deleted successfully');
      } catch (error) {
        console.error("Error deleting IP:", error);
        message.error(`Failed to delete IP address ${error}`);
      } finally {
        setIsDeleteIPModalVisible(false);
        setIPToDelete(null);
      }
    }
  };


  const handleAddSSOOk = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
  };

  const handleAddSSOCancel = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
  };

  const handleShowInstructions = (formValues: Record<string, any>) => {
    handleAdminCreate(formValues);
    handleSSOUpdate(formValues);
    setIsAddSSOModalVisible(false);
    setIsInstructionsModalVisible(true);
    // Optionally, you can call handleSSOUpdate here with the formValues
  };

  const handleInstructionsOk = () => {
    setIsInstructionsModalVisible(false);
  };

  const handleInstructionsCancel = () => {
    setIsInstructionsModalVisible(false);
  };

  const roles = ["proxy_admin", "proxy_admin_viewer"];

  useEffect(() => {
    if (router) {
      const { protocol, host } = window.location;
      const baseUrl = `${protocol}//${host}`;
      setBaseUrl(baseUrl);
    }
  }, [router]);

  useEffect(() => {
    // Fetch model info and set the default selected model
    const fetchProxyAdminInfo = async () => {
      if (accessToken != null) {
        const combinedList: any[] = [];
        const response = await userGetAllUsersCall(
          accessToken,
          "proxy_admin_viewer"
        );
        console.log("proxy admin viewer response: ", response);
        const proxyViewers: User[] = response["users"];
        console.log(`proxy viewers response: ${proxyViewers}`);
        proxyViewers.forEach((viewer: User) => {
          combinedList.push({
            user_role: viewer.user_role,
            user_id: viewer.user_id,
            user_email: viewer.user_email,
          });
        });

        console.log(`proxy viewers: ${proxyViewers}`);

        const response2 = await userGetAllUsersCall(
          accessToken,
          "proxy_admin"
        );

        const proxyAdmins: User[] = response2["users"];

        proxyAdmins.forEach((admins: User) => {
          combinedList.push({
            user_role: admins.user_role,
            user_id: admins.user_id,
            user_email: admins.user_email,
          });
        });

        console.log(`proxy admins: ${proxyAdmins}`);
        console.log(`combinedList: ${combinedList}`);
        setAdmins(combinedList);

        const availableUserRoles = await getPossibleUserRoles(accessToken);
        setPossibleUIRoles(availableUserRoles);
      }
    };

    fetchProxyAdminInfo();
  }, [accessToken]);

  const handleMemberUpdateOk = () => {
    setIsUpdateModalModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };

  const handleAdminOk = () => {
    setIsAddAdminModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };

  const handleAdminCancel = () => {
    setIsAddAdminModalVisible(false);
    setIsInvitationLinkModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };

  const handleMemberUpdateCancel = () => {
    setIsUpdateModalModalVisible(false);
    memberForm.resetFields();
    form.resetFields();
  };
  // Define the type for the handleMemberCreate function
  type HandleMemberCreate = (formValues: Record<string, any>) => Promise<void>;

  const addMemberForm = (handleMemberCreate: HandleMemberCreate) => {
    return (
      <Form
        form={form}
        onFinish={handleMemberCreate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
      >
        <>
          <Form.Item label="Email" name="user_email" className="mb-8 mt-4">
            <Input
              name="user_email"
              className="px-3 py-2 border rounded-md w-full"
            />
          </Form.Item>
          {/* <div className="text-center mb-4">OR</div>
          <Form.Item label="User ID" name="user_id" className="mb-4">
            <Input
              name="user_id"
              className="px-3 py-2 border rounded-md w-full"
            />
          </Form.Item> */}
        </>
        <div style={{ textAlign: "right", marginTop: "10px" }} className="mt-4">
          <Button2 htmlType="submit">Add member</Button2>
        </div>
      </Form>
    );
  };

  const modifyMemberForm = (
    handleMemberUpdate: HandleMemberCreate,
    currentRole: string,
    userID: string
  ) => {
    return (
      <Form
        form={form}
        onFinish={handleMemberUpdate}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
      >
        <>
          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="User Role"
            name="user_role"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <Select value={currentRole}>
              {roles.map((role, index) => (
                <SelectItem key={index} value={role}>
                  {role}
                </SelectItem>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            label="Team ID"
            name="user_id"
            hidden={true}
            initialValue={userID}
            valuePropName="user_id"
            className="mt-8"
          >
            <Input value={userID} disabled />
          </Form.Item>
        </>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 htmlType="submit">Update role</Button2>
        </div>
      </Form>
    );
  };

  const handleMemberUpdate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && admins != null) {
        message.info("Making API Call");
        const response: any = await userUpdateUserCall(
          accessToken,
          formValues,
          null
        );
        console.log(`response for team create call: ${response}`);
        // Checking if the team exists in the list and updating or adding accordingly
        const foundIndex = admins.findIndex((user) => {
          console.log(
            `user.user_id=${user.user_id}; response.user_id=${response.user_id}`
          );
          return user.user_id === response.user_id;
        });
        console.log(`foundIndex: ${foundIndex}`);
        if (foundIndex == -1) {
          console.log(`updates admin with new user`);
          admins.push(response);
          // If new user is found, update it
          setAdmins(admins); // Set the new state
        }
        message.success("Refresh tab to see updated user role");
        setIsUpdateModalModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && admins != null) {
        message.info("Making API Call");
        const response: any = await userUpdateUserCall(
          accessToken,
          formValues,
          "proxy_admin_viewer"
        );
        console.log(`response for team create call: ${response}`);
        // Checking if the team exists in the list and updating or adding accordingly

        // Give admin an invite link for inviting user to proxy
        const user_id = response.data?.user_id || response.user_id;
        invitationCreateCall(accessToken, user_id).then((data) => {
          setInvitationLinkData(data);
          setIsInvitationLinkModalVisible(true);
        });

        const foundIndex = admins.findIndex((user) => {
          console.log(
            `user.user_id=${user.user_id}; response.user_id=${response.user_id}`
          );
          return user.user_id === response.user_id;
        });
        console.log(`foundIndex: ${foundIndex}`);
        if (foundIndex == -1) {
          console.log(`updates admin with new user`);
          admins.push(response);
          // If new user is found, update it
          setAdmins(admins); // Set the new state
        }
        form.resetFields();
        setIsAddMemberModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };
  const handleAdminCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && admins != null) {
        message.info("Making API Call");
        const user_role: Member = {
          role: "user",
          user_email: formValues.user_email,
          user_id: formValues.user_id,
        };
        const response: any = await userUpdateUserCall(
          accessToken,
          formValues,
          "proxy_admin"
        );

        // Give admin an invite link for inviting user to proxy
        const user_id = response.data?.user_id || response.user_id;
        invitationCreateCall(accessToken, user_id).then((data) => {
          setInvitationLinkData(data);
          setIsInvitationLinkModalVisible(true);
        });
        console.log(`response for team create call: ${response}`);
        // Checking if the team exists in the list and updating or adding accordingly
        const foundIndex = admins.findIndex((user) => {
          console.log(
            `user.user_id=${user.user_id}; response.user_id=${user_id}`
          );
          return user.user_id === response.user_id;
        });
        console.log(`foundIndex: ${foundIndex}`);
        if (foundIndex == -1) {
          console.log(`updates admin with new user`);
          admins.push(response);
          // If new user is found, update it
          setAdmins(admins); // Set the new state
        }
        form.resetFields();
        setIsAddAdminModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const handleSSOUpdate = async (formValues: Record<string, any>) => {
    if (accessToken == null) {
      return;
    }
    let payload = {
      environment_variables: {
        PROXY_BASE_URL: formValues.proxy_base_url,
        GOOGLE_CLIENT_ID: formValues.google_client_id,
        GOOGLE_CLIENT_SECRET: formValues.google_client_secret,
      },
    };
    setCallbacksCall(accessToken, payload);
  };
  console.log(`admins: ${admins?.length}`);
  return (
    <div className="w-full m-2 mt-2 p-8">
      <Title level={4}>Admin Access </Title>
      <Paragraph>
        {showSSOBanner && (
          <a href="https://docs.litellm.ai/docs/proxy/ui#restrict-ui-access">
            Requires SSO Setup
          </a>
        )}
        <br />
        <b>Proxy Admin: </b> Can create keys, teams, users, add models, etc.{" "}
        <br />
        <b>Proxy Admin Viewer: </b>Can just view spend. They cannot create keys,
        teams or grant users access to new models.{" "}
      </Paragraph>
      <Grid numItems={1} className="gap-2 p-2 w-full">
        <Col numColSpan={1}>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Member Name</TableHeaderCell>
                  <TableHeaderCell>Role</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                {admins
                  ? admins.map((member: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>
                          {member["user_email"]
                            ? member["user_email"]
                            : member["user_id"]
                              ? member["user_id"]
                              : null}
                        </TableCell>
                        <TableCell>
                          {" "}
                          {possibleUIRoles?.[member?.user_role]?.ui_label ||
                            "-"}
                        </TableCell>
                        <TableCell>
                          <Icon
                            icon={PencilAltIcon}
                            size="sm"
                            onClick={() => setIsUpdateModalModalVisible(true)}
                          />
                          <Modal
                            title="Update role"
                            visible={isUpdateMemberModalVisible}
                            width={800}
                            footer={null}
                            onOk={handleMemberUpdateOk}
                            onCancel={handleMemberUpdateCancel}
                          >
                            {modifyMemberForm(
                              handleMemberUpdate,
                              member["user_role"],
                              member["user_id"]
                            )}
                          </Modal>
                        </TableCell>
                      </TableRow>
                    ))
                  : null}
              </TableBody>
            </Table>
          </Card>
        </Col>
        <Col numColSpan={1}>
          <div className="flex justify-start">
            <Button
              className="mr-4 mb-5"
              onClick={() => setIsAddAdminModalVisible(true)}
            >
              + Add admin
            </Button>
            <Modal
              title="Add admin"
              visible={isAddAdminModalVisible}
              width={800}
              footer={null}
              onOk={handleAdminOk}
              onCancel={handleAdminCancel}
            >
              {addMemberForm(handleAdminCreate)}
            </Modal>
            <OnboardingModal
              isInvitationLinkModalVisible={isInvitationLinkModalVisible}
              setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
              baseUrl={baseUrl}
              invitationLinkData={invitationLinkData}
            />
            <Button
              className="mb-5"
              onClick={() => setIsAddMemberModalVisible(true)}
            >
              + Add viewer
            </Button>
            <Modal
              title="Add viewer"
              visible={isAddMemberModalVisible}
              width={800}
              footer={null}
              onOk={handleMemberOk}
              onCancel={handleMemberCancel}
            >
              {addMemberForm(handleMemberCreate)}
            </Modal>
          </div>
        </Col>
      </Grid>
      <Grid >
        <Card>
        <Title level={4}> ✨ Security Settings</Title>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
    <div>
      <Button onClick={() => premiumUser === true ? setIsAddSSOModalVisible(true) : message.error("Only premium users can add SSO")}>Add SSO</Button>
    </div>
    <div>
      <Button onClick={handleShowAllowedIPs}>Allowed IPs</Button>
    </div>
  </div>
        </Card>
       
        <div className="flex justify-start mb-4">
         
          <Modal
            title="Add SSO"
            visible={isAddSSOModalVisible}
            width={800}
            footer={null}
            onOk={handleAddSSOOk}
            onCancel={handleAddSSOCancel}
          >
            <Form
              form={form}
              onFinish={handleShowInstructions}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <>
                <Form.Item
                  label="Admin Email"
                  name="user_email"
                  rules={[
                    {
                      required: true,
                      message: "Please enter the email of the proxy admin",
                    },
                  ]}
                >
                  <Input />
                </Form.Item>
                <Form.Item
                  label="PROXY BASE URL"
                  name="proxy_base_url"
                  rules={[
                    {
                      required: true,
                      message: "Please enter the proxy base url",
                    },
                  ]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label="GOOGLE CLIENT ID"
                  name="google_client_id"
                  rules={[
                    {
                      required: true,
                      message: "Please enter the google client id",
                    },
                  ]}
                >
                  <Input.Password />
                </Form.Item>

                <Form.Item
                  label="GOOGLE CLIENT SECRET"
                  name="google_client_secret"
                  rules={[
                    {
                      required: true,
                      message: "Please enter the google client secret",
                    },
                  ]}
                >
                  <Input.Password />
                </Form.Item>
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Save</Button2>
              </div>
            </Form>
          </Modal>
          <Modal
            title="SSO Setup Instructions"
            visible={isInstructionsModalVisible}
            width={800}
            footer={null}
            onOk={handleInstructionsOk}
            onCancel={handleInstructionsCancel}
          >
            <p>Follow these steps to complete the SSO setup:</p>
            <Text className="mt-2">1. DO NOT Exit this TAB</Text>
            <Text className="mt-2">
              2. Open a new tab, visit your proxy base url
            </Text>
            <Text className="mt-2">
              3. Confirm your SSO is configured correctly and you can login on
              the new Tab
            </Text>
            <Text className="mt-2">
              4. If Step 3 is successful, you can close this tab
            </Text>
            <div style={{ textAlign: "right", marginTop: "10px" }}>
              <Button2 onClick={handleInstructionsOk}>Done</Button2>
            </div>
          </Modal>
          <Modal
          title="Manage Allowed IP Addresses"
          width={800}
          visible={isAllowedIPModalVisible}
          onCancel={() => setIsAllowedIPModalVisible(false)}
          footer={[
            <Button className="mx-1"key="add" onClick={() => setIsAddIPModalVisible(true)}>
              Add IP Address
            </Button>,
            <Button key="close" onClick={() => setIsAllowedIPModalVisible(false)}>
              Close
            </Button>
          ]}
        >
          <Table>
  <TableHead>
    <TableRow>
      <TableHeaderCell>IP Address</TableHeaderCell>
      <TableHeaderCell className="text-right">Action</TableHeaderCell>
    </TableRow>
  </TableHead>
  <TableBody>
  {allowedIPs.map((ip, index) => (
  <TableRow key={index}>
    <TableCell>{ip}</TableCell>
    <TableCell className="text-right">
      {ip !== all_ip_address_allowed && (
        <Button onClick={() => handleDeleteIP(ip)} color="red" size="xs">
          Delete
        </Button>
      )}
    </TableCell>
  </TableRow>
))}
  </TableBody>
</Table>
        </Modal>

        <Modal
          title="Add Allowed IP Address"
          visible={isAddIPModalVisible}
          onCancel={() => setIsAddIPModalVisible(false)}
          footer={null}
        >
          <Form onFinish={handleAddIP}>
            <Form.Item
              name="ip"
              rules={[{ required: true, message: 'Please enter an IP address' }]}
            >
              <Input placeholder="Enter IP address" />
            </Form.Item>
            <Form.Item>
              <Button2 htmlType="submit">
                Add IP Address
              </Button2>
            </Form.Item>
          </Form>
        </Modal>

        <Modal
          title="Confirm Delete"
          visible={isDeleteIPModalVisible}
          onCancel={() => setIsDeleteIPModalVisible(false)}
          onOk={confirmDeleteIP}
          footer={[
            <Button className="mx-1"key="delete" onClick={() => confirmDeleteIP()}>
              Yes
            </Button>,
            <Button key="close" onClick={() => setIsDeleteIPModalVisible(false)}>
              Close
            </Button>
          ]}
        >
          <p>Are you sure you want to delete the IP address: {ipToDelete}?</p>
        </Modal>
        </div>
        <Callout title="Login without SSO" color="teal">
          If you need to login without sso, you can access{" "}
          <a href={nonSssoUrl} target="_blank">
            <b>{nonSssoUrl}</b>{" "}
          </a>
        </Callout>
      </Grid>
    </div>
  );
};

export default AdminPanel;
