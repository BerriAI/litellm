import React, { useState } from "react";
import { Button as Button2, Text } from "@tremor/react";
import { Modal, Table, Upload, message } from "antd";
import { UploadOutlined, DownloadOutlined } from "@ant-design/icons";
import { userCreateCall } from "./networking";
import Papa from "papaparse";

interface BulkCreateUsersProps {
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
}

interface UserData {
  user_email: string;
  user_role: string;
  team_id?: string;
  metadata?: string;
}

const BulkCreateUsers: React.FC<BulkCreateUsersProps> = ({
  accessToken,
  teams,
  possibleUIRoles,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [parsedData, setParsedData] = useState<UserData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const downloadTemplate = () => {
    const template = [
      ["user_email", "user_role", "team_id", "metadata"],
      ["user@example.com", "internal_user", "", "{}"],
    ];
    
    const csv = Papa.unparse(template);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bulk_users_template.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const handleFileUpload = (file: File) => {
    Papa.parse(file, {
      complete: (results) => {
        const headers = results.data[0] as string[];
        const userData = results.data.slice(1).map((row: any) => {
          const user: UserData = {
            user_email: row[headers.indexOf("user_email")],
            user_role: row[headers.indexOf("user_role")],
            team_id: row[headers.indexOf("team_id")],
            metadata: row[headers.indexOf("metadata")],
          };
          return user;
        });
        setParsedData(userData.filter(user => user.user_email && user.user_role));
      },
      header: false,
    });
    return false;
  };

  const handleBulkCreate = async () => {
    setIsProcessing(true);
    const results = {
      success: 0,
      failed: 0,
      errors: [] as string[],
    };

    for (const user of parsedData) {
      try {
        await userCreateCall(accessToken, null, user);
        results.success++;
      } catch (error) {
        results.failed++;
        results.errors.push(`Failed to create user ${user.user_email}: ${error}`);
      }
    }

    message.info(`Created ${results.success} users. Failed: ${results.failed}`);
    if (results.errors.length > 0) {
      console.error("Errors:", results.errors);
    }

    setIsProcessing(false);
    setIsModalVisible(false);
    setParsedData([]);
  };

  const columns = [
    {
      title: "Email",
      dataIndex: "user_email",
      key: "user_email",
    },
    {
      title: "Role",
      dataIndex: "user_role",
      key: "user_role",
    },
    {
      title: "Team",
      dataIndex: "team_id",
      key: "team_id",
    },
  ];

  return (
    <>
      <Button2 className="mx-auto mb-0" onClick={() => setIsModalVisible(true)}>
        + Bulk Invite Users
      </Button2>
      
      <Modal
        title={`Bulk Create Users (${parsedData.length} users)`}
        visible={isModalVisible}
        width={800}
        onCancel={() => setIsModalVisible(false)}
        bodyStyle={{ maxHeight: '70vh', overflow: 'auto' }}
        footer={[
          <Button2 key="download" onClick={downloadTemplate}>
            <DownloadOutlined /> Download Template
          </Button2>,
          <Button2
            key="submit"
            onClick={handleBulkCreate}
            disabled={parsedData.length === 0 || isProcessing}
          >
            Create {parsedData.length} Users
          </Button2>,
        ]}
      >
          
          <Text>Upload a CSV file with user information</Text>
          
          <Upload
            beforeUpload={handleFileUpload}
            accept=".csv"
            maxCount={1}
            showUploadList={false}
          >
            <Button2>
              <UploadOutlined className="mr-2" /> Upload CSV
            </Button2>
          </Upload>

          {parsedData.length > 0 && (
            <div>
              <Text>Preview of users to be created:</Text>
              <Table
                dataSource={parsedData}
                columns={columns}
                size="small"
                pagination={{ pageSize: 5 }}
                scroll={{ y: 300 }}
              />
            </div>
          )}
      </Modal>
    </>
  );
};

export default BulkCreateUsers; 