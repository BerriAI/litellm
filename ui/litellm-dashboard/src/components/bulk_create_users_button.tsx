import React, { useState } from "react";
import { Button as Button2, Text } from "@tremor/react";
import { Modal, Table, Upload, message } from "antd";
import { UploadOutlined, DownloadOutlined } from "@ant-design/icons";
import { userCreateCall } from "./networking";
import Papa from "papaparse";
import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/outline";

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
  status?: string;
  error?: string;
  rowNumber?: number;
  isValid?: boolean;
}

const BulkCreateUsers: React.FC<BulkCreateUsersProps> = ({
  accessToken,
  teams,
  possibleUIRoles,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [parsedData, setParsedData] = useState<UserData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

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
    setParseError(null);
    Papa.parse(file, {
      complete: (results) => {
        const headers = results.data[0] as string[];
        const requiredColumns = ['user_email', 'user_role'];
        
        // Check if all required columns are present
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        if (missingColumns.length > 0) {
          setParseError(`Your CSV is missing these required columns: ${missingColumns.join(', ')}`);
          setParsedData([]);
          return;
        }

        try {
          const userData = results.data.slice(1).map((row: any, index: number) => {
            const user: UserData = {
              user_email: row[headers.indexOf("user_email")]?.trim() || '',
              user_role: row[headers.indexOf("user_role")]?.trim() || '',
              team_id: row[headers.indexOf("team_id")]?.trim(),
              metadata: row[headers.indexOf("metadata")]?.trim(),
              rowNumber: index + 2,
              isValid: true,
              error: '',
            };

            // Validate the row
            const errors: string[] = [];
            if (!user.user_email) errors.push('Email is required');
            if (!user.user_role) errors.push('Role is required');
            if (user.user_email && !user.user_email.includes('@')) errors.push('Invalid email format');

            if (errors.length > 0) {
              user.isValid = false;
              user.error = errors.join(', ');
            }

            return user;
          });

          const validData = userData.filter(user => user.isValid);
          setParsedData(userData);

          if (validData.length === 0) {
            setParseError('No valid users found in the CSV. Please check the errors below.');
          } else if (validData.length < userData.length) {
            setParseError(`Found ${userData.length - validData.length} row(s) with errors. Please correct them before proceeding.`);
          } else {
            message.success(`Successfully parsed ${validData.length} users`);
          }
        } catch (error) {
          setParseError(`Error parsing CSV: ${error.message}`);
          setParsedData([]);
        }
      },
      error: (error) => {
        setParseError(`Failed to parse CSV file: ${error.message}`);
        setParsedData([]);
      },
      header: false,
    });
    return false;
  };

  const handleBulkCreate = async () => {
    setIsProcessing(true);
    const updatedData = parsedData.map(user => ({ ...user, status: 'pending' }));
    setParsedData(updatedData);

    for (const [index, user] of updatedData.entries()) {
      try {
        const response = await userCreateCall(accessToken, null, user);
        
        if (response?.status === 200) {
          setParsedData(current => 
            current.map((u, i) => 
              i === index ? { ...u, status: 'success' } : u
            )
          );
        } else {
          const errorMessage = response?.error?.message?.error;
          setParsedData(current => 
            current.map((u, i) => 
              i === index ? { ...u, status: 'failed', error: errorMessage } : u
            )
          );
        }
      } catch (error) {
        const errorMessage = error?.response?.data?.error?.message?.error || 'Failed to create user';
        setParsedData(current => 
          current.map((u, i) => 
            i === index ? { ...u, status: 'failed', error: errorMessage } : u
          )
        );
      }
    }

    setIsProcessing(false);
  };

  const columns = [
    {
      title: "Row",
      dataIndex: "rowNumber",
      key: "rowNumber",
      width: 80,
    },
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
    {
      title: 'Status',
      key: 'status',
      render: (_, record: UserData) => {
        if (!record.isValid) {
          return (
            <div>
              <div className="flex items-center">
                <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-red-500">Invalid</span>
              </div>
              {record.error && (
                <span className="text-sm text-red-500 ml-7">{record.error}</span>
              )}
            </div>
          );
        }
        if (!record.status || record.status === 'pending') {
          return <span className="text-gray-500">Pending</span>;
        }
        if (record.status === 'success') {
          return (
            <div className="flex items-center">
              <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
              <span className="text-green-500">Success</span>
            </div>
          );
        }
        return (
          <div>
            <div className="flex items-center">
              <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-red-500">Failed</span>
            </div>
            {record.error && (
              <span className="text-sm text-red-500 ml-7">
                {JSON.stringify(record.error)}
              </span>
            )}
          </div>
        );
      },
    },
  ];

  return (
    <>
      <Button2 className="mx-auto mb-0" onClick={() => setIsModalVisible(true)}>
        + Bulk Invite Users
      </Button2>
      
      <Modal
        title="Bulk Invite Users"
        visible={isModalVisible}
        width={800}
        onCancel={() => setIsModalVisible(false)}
        bodyStyle={{ maxHeight: '70vh', overflow: 'auto' }}
        footer={null}
      >
        <div className="flex flex-col">
          {/* Step indicator */}
          {parsedData.length === 0 ? (
            <div className="mb-6">
              <div className="flex items-center mb-4">
                <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">1</div>
                <h3 className="text-lg font-medium">Download and fill the template</h3>
              </div>
              
              <div className="ml-11 mb-6">
                <p className="mb-4">Add multiple users at once by following these steps:</p>
                <ol className="list-decimal list-inside space-y-2 ml-2 mb-4">
                  <li>Download our CSV template</li>
                  <li>Add your users' information to the spreadsheet</li>
                  <li>Save the file and upload it here</li>
                </ol>
                
                <div className="bg-gray-50 p-4 rounded-md border border-gray-200 mb-4">
                  <h4 className="font-medium mb-2">Template fields:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">user_email</p>
                        <p className="text-sm text-gray-600">User's email address (required)</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">user_role</p>
                        <p className="text-sm text-gray-600">User's role (e.g., "internal_user")</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">team_id</p>
                        <p className="text-sm text-gray-600">Optional team assignment</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">metadata</p>
                        <p className="text-sm text-gray-600">Optional JSON metadata (use {} for empty)</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Button2 
                  onClick={downloadTemplate} 
                  size="lg" 
                  className="w-full md:w-auto"
                >
                  <DownloadOutlined className="mr-2" /> Download CSV Template
                </Button2>
              </div>
              
              <div className="flex items-center mb-4">
                <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">2</div>
                <h3 className="text-lg font-medium">Upload your completed CSV</h3>
              </div>
              
              <div className="ml-11">
                <Upload
                  beforeUpload={handleFileUpload}
                  accept=".csv"
                  maxCount={1}
                  showUploadList={false}
                >
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer">
                    <UploadOutlined className="text-3xl text-gray-400 mb-2" />
                    <p className="mb-1">Drag and drop your CSV file here</p>
                    <p className="text-sm text-gray-500 mb-3">or</p>
                    <Button2 size="sm">Browse files</Button2>
                  </div>
                </Upload>
              </div>
            </div>
          ) : (
            <div className="mb-6">
              <div className="flex items-center mb-4">
                <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">3</div>
                <h3 className="text-lg font-medium">Review and create users</h3>
              </div>
              
              {parseError && (
                <div className="ml-11 mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
                  <Text className="text-red-600 font-medium">{parseError}</Text>
                </div>
              )}
              
              <div className="ml-11">
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center">
                    <Text className="text-lg font-medium mr-3">User Preview</Text>
                    <Text className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                      {parsedData.filter(d => d.isValid).length} of {parsedData.length} users valid
                    </Text>
                  </div>
                  
                  <div className="flex space-x-3">
                    <Button2 
                      onClick={() => {
                        setParsedData([]);
                        setParseError(null);
                      }} 
                      variant="secondary"
                    >
                      Back
                    </Button2>
                    <Button2
                      onClick={handleBulkCreate}
                      disabled={parsedData.filter(d => d.isValid).length === 0 || isProcessing}
                    >
                      {isProcessing ? 'Creating...' : `Create ${parsedData.filter(d => d.isValid).length} Users`}
                    </Button2>
                  </div>
                </div>
                
                <Table
                  dataSource={parsedData}
                  columns={columns}
                  size="small"
                  pagination={{ pageSize: 5 }}
                  scroll={{ y: 300 }}
                  rowClassName={(record) => !record.isValid ? 'bg-red-50' : ''}
                />
                
                <div className="flex justify-end mt-4">
                  <Button2 
                    onClick={() => {
                      setParsedData([]);
                      setParseError(null);
                    }} 
                    variant="secondary"
                    className="mr-3"
                  >
                    Back
                  </Button2>
                  <Button2
                    onClick={handleBulkCreate}
                    disabled={parsedData.filter(d => d.isValid).length === 0 || isProcessing}
                  >
                    {isProcessing ? 'Creating...' : `Create ${parsedData.filter(d => d.isValid).length} Users`}
                  </Button2>
                </div>
              </div>
            </div>
          )}
        </div>
      </Modal>
    </>
  );
};

export default BulkCreateUsers; 