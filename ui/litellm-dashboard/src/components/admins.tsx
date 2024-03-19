/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
import React, { useState, useEffect } from "react";
import { Typography } from "antd";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
} from "antd";
import { Select, SelectItem } from "@tremor/react";
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
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
interface AdminPanelProps {
  searchParams: any;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
}
import {
  userUpdateUserCall,
  Member,
  userGetAllUsersCall,
  User,
} from "./networking";

const AdminPanel: React.FC<AdminPanelProps> = ({
  searchParams,
  accessToken,
}) => {
  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  const [admins, setAdmins] = useState<null | any[]>(null);

  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);

  useEffect(() => {
    // Fetch model info and set the default selected model
    const fetchProxyAdminInfo = async () => {
      if (accessToken != null) {
        const combinedList: any[] = [];
        const proxyViewers = await userGetAllUsersCall(
          accessToken,
          "proxy_admin_viewer"
        );
        proxyViewers.forEach((viewer: User) => {
          combinedList.push({
            user_role: viewer.user_role,
            user_id: viewer.user_id,
            user_email: viewer.user_email,
          });
        });

        console.log(`proxy viewers: ${proxyViewers}`);

        const proxyAdmins = await userGetAllUsersCall(
          accessToken,
          "proxy_admin"
        );

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
      }
    };

    fetchProxyAdminInfo();
  }, [accessToken]);

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      if (accessToken != null && admins != null) {
        message.info("Making API Call");
        const user_role: Member = {
          role: "user",
          user_email: formValues.user_email,
          user_id: formValues.user_id,
        };
        const response: any = await userUpdateUserCall(accessToken, formValues);
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
        setIsAddMemberModalVisible(false);
      }
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };
  console.log(`admins: ${admins?.length}`);
  return (
    <div className="w-full m-2">
      <Title level={4}>Restricted Access</Title>
      <Paragraph>
        Add other people to just view spend. They cannot create keys, teams or
        grant users access to new models.{" "}
        <a href="https://docs.litellm.ai/docs/proxy/ui#restrict-ui-access">
          Requires SSO Setup
        </a>
      </Paragraph>
      <Grid numItems={1} className="gap-2 p-0 w-full">
        <Col numColSpan={1}>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Member Name</TableHeaderCell>
                  <TableHeaderCell>Role</TableHeaderCell>
                  <TableHeaderCell>Action</TableHeaderCell>
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
                        <TableCell>{member["user_role"]}</TableCell>
                        <TableCell>
                          <Icon icon={CogIcon} size="sm" />
                        </TableCell>
                      </TableRow>
                    ))
                  : null}
              </TableBody>
            </Table>
          </Card>
        </Col>
        <Col numColSpan={1}>
          <Button
            className="mx-auto mb-5"
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
            <Form
              form={form}
              onFinish={handleMemberCreate}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <>
                <Form.Item label="Email" name="user_email" className="mb-4">
                  <Input
                    name="user_email"
                    className="px-3 py-2 border rounded-md w-full"
                  />
                </Form.Item>
                <div className="text-center mb-4">OR</div>
                <Form.Item label="User ID" name="user_id" className="mb-4">
                  <Input
                    name="user_id"
                    className="px-3 py-2 border rounded-md w-full"
                  />
                </Form.Item>
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Add member</Button2>
              </div>
            </Form>
          </Modal>
        </Col>
      </Grid>
    </div>
  );
};

export default AdminPanel;
