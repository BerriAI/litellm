import React, { useState, useEffect } from "react";
import Link from "next/link";
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
  Grid,
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";
interface TeamProps {
  teams: string[] | null;
  searchParams: any;
}

const Team: React.FC<TeamProps> = ({ teams, searchParams }) => {
  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();
  const { Title, Paragraph } = Typography;
  const [value, setValue] = useState("");
  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
  };

  const handleMemberCancel = () => {
    setIsAddMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      console.log("reaches here");
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const handleMemberCreate = async (formValues: Record<string, any>) => {
    try {
      console.log("reaches here");
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };
  console.log(`received teams ${teams}`);
  return (
    <div className="w-full">
      <Grid numItems={1} className="gap-2 p-2 h-[75vh] w-full">
        <Col numColSpan={1}>
          <Title level={4}>All Teams</Title>
          <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Team Name</TableHeaderCell>
                  <TableHeaderCell>Spend (USD)</TableHeaderCell>
                  <TableHeaderCell>Budget (USD)</TableHeaderCell>
                  <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                </TableRow>
              </TableHead>

              <TableBody>
                <TableRow>
                  <TableCell>Wilhelm Tell</TableCell>
                  <TableCell className="text-right">1</TableCell>
                  <TableCell>Uri, Schwyz, Unterwalden</TableCell>
                  <TableCell>National Hero</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>The Witcher</TableCell>
                  <TableCell className="text-right">129</TableCell>
                  <TableCell>Kaedwen</TableCell>
                  <TableCell>Legend</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Mizutsune</TableCell>
                  <TableCell className="text-right">82</TableCell>
                  <TableCell>Japan</TableCell>
                  <TableCell>N/A</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </Card>
        </Col>
        <Col numColSpan={1}>
          <Button
            className="mx-auto"
            onClick={() => setIsTeamModalVisible(true)}
          >
            + Create New Team
          </Button>
          <Modal
            title="Create Team"
            visible={isTeamModalVisible}
            width={800}
            footer={null}
            onOk={handleOk}
            onCancel={handleCancel}
          >
            <Form
              form={form}
              onFinish={handleCreate}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <>
                <Form.Item label="Team Name" name="team_alias">
                  <Input />
                </Form.Item>
                <Form.Item label="Models" name="models">
                  <Select2
                    mode="multiple"
                    placeholder="Select models"
                    style={{ width: "100%" }}
                  >
                    {/* {userModels.map((model) => (
                      <Option key={model} value={model}>
                        {model}
                      </Option>
                    ))} */}
                  </Select2>
                </Form.Item>
                <Form.Item label="Max Budget (USD)" name="max_budget">
                  <InputNumber step={0.01} precision={2} width={200} />
                </Form.Item>
                <Form.Item
                  label="Tokens per minute Limit (TPM)"
                  name="tpm_limit"
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
                <Form.Item
                  label="Requests per minute Limit (RPM)"
                  name="rpm_limit"
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">Create Team</Button2>
              </div>
            </Form>
          </Modal>
        </Col>
        <Col numColSpan={1}>
          <Title level={4}>Team Members</Title>
          <Paragraph>
            If you belong to multiple teams, this setting controls which teams'
            members you see.
          </Paragraph>
          {teams && teams.length > 0 ? (
            <Select
              id="distance"
              value={value}
              onValueChange={setValue}
              className="mt-2"
            >
              {teams.map((model) => (
                <SelectItem value="model">{model}</SelectItem>
              ))}
            </Select>
          ) : (
            <Paragraph>
              No team created. <b>Defaulting to personal account.</b>
            </Paragraph>
          )}
        </Col>
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
                <TableRow>
                  <TableCell>Wilhelm Tell</TableCell>
                  <TableCell>Uri, Schwyz, Unterwalden</TableCell>
                  <TableCell>
                    <Icon icon={CogIcon} size="sm" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>The Witcher</TableCell>
                  <TableCell>Kaedwen</TableCell>
                  <TableCell>
                    <Icon icon={CogIcon} size="sm" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Mizutsune</TableCell>
                  <TableCell>Japan</TableCell>
                  <TableCell>
                    <Icon icon={CogIcon} size="sm" />
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </Card>
        </Col>
        <Col numColSpan={1}>
          <Button
            className="mx-auto mb-5"
            onClick={() => setIsAddMemberModalVisible(true)}
          >
            + Add member
          </Button>
          <Modal
            title="Add member"
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
                <Form.Item label="Email" name="user_email">
                  <Input />
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

export default Team;
