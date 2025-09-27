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
import { Team } from "./key_team_helpers/key_list";
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
  TabGroup,
  TabList,
  Tab,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { PencilAltIcon } from "@heroicons/react/outline";
import OnboardingModal from "./onboarding_link";
import { InvitationLink } from "./onboarding_link";
import SSOModals from "./SSOModals";
import { ssoProviderConfigs } from './SSOModals';
import SCIMConfig from "./SCIM";
import UIAccessControlForm from "./UIAccessControlForm";
import NotificationsManager from "./molecules/notifications_manager";

interface AdminPanelProps {
  searchParams: any;
  accessToken: string | null;
  userID: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Team[] | null>>;
  showSSOBanner: boolean;
  premiumUser: boolean;
  proxySettings?: any;
  userRole?: string | null;
}
import { useBaseUrl } from "./constants";


import {
  userUpdateUserCall,
  Member,
  userGetAllUsersCall,
  User,
  invitationCreateCall,
  getPossibleUserRoles,
  addAllowedIP,
  getAllowedIPs,
  deleteAllowedIP,
  getSSOSettings,
} from "./networking";

const AdminPanel: React.FC<AdminPanelProps> = ({
  searchParams,
  accessToken,
  userID,
  showSSOBanner,
  premiumUser,
  proxySettings,
  userRole,
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
  const [isUIAccessControlModalVisible, setIsUIAccessControlModalVisible] = useState(false);
  const [allowedIPs, setAllowedIPs] = useState<string[]>([]);
  const [ipToDelete, setIPToDelete] = useState<string | null>(null);
  const [ssoConfigured, setSsoConfigured] = useState<boolean>(false);
  const router = useRouter();

  const [possibleUIRoles, setPossibleUIRoles] = useState<null | Record<
    string,
    Record<string, string>
  >>(null);

  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function() {};
  }

  const baseUrl = useBaseUrl();
  const all_ip_address_allowed = "All IP Addresses Allowed";

  let nonSssoUrl = baseUrl;
  nonSssoUrl += "/fallback/login";

  // Extract the SSO configuration check logic into a separate function for reuse
  const checkSSOConfiguration = async () => {
    if (accessToken && premiumUser) {
      try {
        const ssoData = await getSSOSettings(accessToken);
        console.log("SSO data:", ssoData);
        
        // Check if any SSO provider is configured
        if (ssoData && ssoData.values) {
          const hasGoogleSSO = ssoData.values.google_client_id && ssoData.values.google_client_secret;
          const hasMicrosoftSSO = ssoData.values.microsoft_client_id && ssoData.values.microsoft_client_secret;
          const hasGenericSSO = ssoData.values.generic_client_id && ssoData.values.generic_client_secret;
          
          setSsoConfigured(hasGoogleSSO || hasMicrosoftSSO || hasGenericSSO);
        } else {
          setSsoConfigured(false);
        }
      } catch (error) {
        console.error("Error checking SSO configuration:", error);
        setSsoConfigured(false);
      }
    }
  };

  const handleShowAllowedIPs = async () => {
    try {
      if (premiumUser !== true) {
        NotificationsManager.fromBackend(
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
      NotificationsManager.fromBackend(`Failed to fetch allowed IPs ${error}`);
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
        NotificationsManager.success('IP address added successfully');
      }
    } catch (error) {
      console.error("Error adding IP:", error);
      NotificationsManager.fromBackend(`Failed to add IP address ${error}`);
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
        NotificationsManager.success('IP address deleted successfully');
      } catch (error) {
        console.error("Error deleting IP:", error);
        NotificationsManager.fromBackend(`Failed to delete IP address ${error}`);
      } finally {
        setIsDeleteIPModalVisible(false);
        setIPToDelete(null);
      }
    }
  };


  const handleAddSSOOk = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
    // Refresh SSO configuration status
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleAddSSOCancel = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
  };

  const handleShowInstructions = (formValues: Record<string, any>) => {
    setIsAddSSOModalVisible(false);
    setIsInstructionsModalVisible(true);
  };

  const handleInstructionsOk = () => {
    setIsInstructionsModalVisible(false);
    // Refresh SSO configuration status after instructions are closed
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleInstructionsCancel = () => {
    setIsInstructionsModalVisible(false);
    // Refresh SSO configuration status after instructions are closed
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const roles = ["proxy_admin", "proxy_admin_viewer"];

  // useEffect(() => {
  //   if (router) {
  //     const { protocol, host } = window.location;
  //     const baseUrl = `${protocol}//${host}`;
  //     setBaseUrl(baseUrl);
  //   }
  // }, [router]);

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

  // Add new useEffect to check SSO configuration
  useEffect(() => {
    checkSSOConfiguration();
  }, [accessToken, premiumUser]);

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
        NotificationsManager.info("Making API Call");
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
        NotificationsManager.success("Refresh tab to see updated user role");
        setIsUpdateModalModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && admins != null) {
        NotificationsManager.info("Making API Call");
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
        NotificationsManager.info("Making API Call");
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

  const handleUIAccessControlOk = () => {
    setIsUIAccessControlModalVisible(false);
  };

  const handleUIAccessControlCancel = () => {
    setIsUIAccessControlModalVisible(false);
  };

  console.log(`admins: ${admins?.length}`);
  return (
    <div className="w-full m-2 mt-2 p-8">
      <Title level={4}>Admin Access </Title>
      <Paragraph>Go to &apos;Internal Users&apos; page to add other admins.</Paragraph>
      <TabGroup>
        <TabList>
          <Tab>Security Settings</Tab>
          <Tab>SCIM</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Card>
              <Title level={4}> ✨ Security Settings</Title>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem', marginLeft: '0.5rem' }}>
                <div>
                  <Button 
                    style={{ width: '150px' }}
                    onClick={() => premiumUser === true ? setIsAddSSOModalVisible(true) : NotificationsManager.fromBackend("Only premium users can add SSO")}
                  >
                    {ssoConfigured ? "Edit SSO Settings" : "Add SSO"}
                  </Button>
                </div>
                <div>
                  <Button 
                    style={{ width: '150px' }}
                    onClick={handleShowAllowedIPs}
                  >
                    Allowed IPs
                  </Button>
                </div>
                <div>
                  <Button 
                    style={{ width: '150px' }}
                    onClick={() => premiumUser === true ? setIsUIAccessControlModalVisible(true) : NotificationsManager.fromBackend("Only premium users can configure UI access control")}
                  >
                    UI Access Control
                  </Button>
                </div>
              </div>
            </Card>
           
            <div className="flex justify-start mb-4">
              <SSOModals
                isAddSSOModalVisible={isAddSSOModalVisible}
                isInstructionsModalVisible={isInstructionsModalVisible}
                handleAddSSOOk={handleAddSSOOk}
                handleAddSSOCancel={handleAddSSOCancel}
                handleShowInstructions={handleShowInstructions}
                handleInstructionsOk={handleInstructionsOk}
                handleInstructionsCancel={handleInstructionsCancel}
                form={form}
                accessToken={accessToken}
                ssoConfigured={ssoConfigured}
              />
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

        {/* UI Access Control Modal */}
        <Modal
          title="UI Access Control Settings"
          visible={isUIAccessControlModalVisible}
          width={600}
          footer={null}
          onOk={handleUIAccessControlOk}
          onCancel={handleUIAccessControlCancel}
        >
          <UIAccessControlForm 
            accessToken={accessToken} 
            onSuccess={() => {
              handleUIAccessControlOk();
              NotificationsManager.success("UI Access Control settings updated successfully");
            }} 
          />
        </Modal>
        </div>
        <Callout title="Login without SSO" color="teal">
          If you need to login without sso, you can access{" "}
          <a href={nonSssoUrl} target="_blank">
            <b>{nonSssoUrl}</b>{" "}
          </a>
        </Callout>
          </TabPanel>
          <TabPanel>
            <SCIMConfig 
              accessToken={accessToken} 
              userID={userID}
              proxySettings={proxySettings}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default AdminPanel;
