import React, { useState, useEffect } from "react";
import { Button as Button2, Text } from "@tremor/react";
import { Modal, Table, Upload, message } from "antd";
import { UploadOutlined, DownloadOutlined } from "@ant-design/icons";
import { userCreateCall, invitationCreateCall, getProxyUISettings } from "./networking";
import Papa from "papaparse";
import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/outline";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { InvitationLink } from "./onboarding_link";

interface BulkCreateUsersProps {
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
}

interface UserData {
  user_email: string;
  user_role: string;
  teams?: string;
  metadata?: string;
  max_budget?: string;
  budget_duration?: string;
  models?: string;
  status?: string;
  error?: string;
  rowNumber?: number;
  isValid?: boolean;
  key?: string;
  invitation_link?: string;
}

// Define an interface for the UI settings
interface UISettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
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
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("http://localhost:4000");

  useEffect(() => {
    // Get UI settings
    const fetchUISettings = async () => {
      try {
        const uiSettingsResponse = await getProxyUISettings(accessToken);
        setUISettings(uiSettingsResponse);
      } catch (error) {
        console.error("Error fetching UI settings:", error);
      }
    };

    fetchUISettings();

    // Set base URL
    const base = new URL("/", window.location.href);
    setBaseUrl(base.toString());
  }, [accessToken]);

  const downloadTemplate = () => {
    const template = [
      ["user_email", "user_role", "teams", "max_budget", "budget_duration", "models"],
      ["user@example.com", "internal_user", "team-id-1,team-id-2", "100", "30d", "gpt-3.5-turbo,gpt-4"],
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
              teams: row[headers.indexOf("teams")]?.trim(),
              max_budget: row[headers.indexOf("max_budget")]?.trim(),
              budget_duration: row[headers.indexOf("budget_duration")]?.trim(),
              models: row[headers.indexOf("models")]?.trim(),
              rowNumber: index + 2,
              isValid: true,
              error: '',
            };

            // Validate the row
            const errors: string[] = [];
            if (!user.user_email) errors.push('Email is required');
            if (!user.user_role) errors.push('Role is required');
            if (user.user_email && !user.user_email.includes('@')) errors.push('Invalid email format');
            
            // Validate user role
            const validRoles = ['proxy_admin', 'proxy_admin_view_only', 'internal_user', 'internal_user_view_only'];
            if (user.user_role && !validRoles.includes(user.user_role)) {
              errors.push(`Invalid role. Must be one of: ${validRoles.join(', ')}`);
            }
            
            // Validate max_budget if provided
            if (user.max_budget && isNaN(parseFloat(user.max_budget))) {
              errors.push('Max budget must be a number');
            }

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
        // Convert teams from comma-separated string to array if provided
        const processedUser = { ...user };
        if (processedUser.teams && processedUser.teams.trim() !== '') {
          processedUser.teams = processedUser.teams.split(',').map(team => team.trim());
        }
        
        // Convert models from comma-separated string to array if provided
        if (processedUser.models && processedUser.models.trim() !== '') {
          processedUser.models = processedUser.models.split(',').map(model => model.trim());
        }
        
        // Convert max_budget to number if provided
        if (processedUser.max_budget && processedUser.max_budget.trim() !== '') {
          processedUser.max_budget = parseFloat(processedUser.max_budget);
        }

        
        const response = await userCreateCall(accessToken, null, processedUser);
        console.log('Full response:', response);
        
        // Check if response has key or user_id, indicating success
        if (response && (response.key || response.user_id)) {
          console.log('Success case triggered');
          const user_id = response.data?.user_id || response.user_id;
          
          // Create invitation link for the user
          try {
            if (!uiSettings?.SSO_ENABLED) {
              // Regular invitation flow
              const invitationData = await invitationCreateCall(accessToken, user_id);
              const invitationUrl = new URL(`/ui?invitation_id=${invitationData.id}`, baseUrl).toString();
              
              setParsedData(current => 
                current.map((u, i) => 
                  i === index ? { 
                    ...u, 
                    status: 'success', 
                    key: response.key || response.user_id,
                    invitation_link: invitationUrl
                  } : u
                )
              );
            } else {
              // SSO flow - just use the base URL
              const invitationUrl = new URL("/ui", baseUrl).toString();
              
              setParsedData(current => 
                current.map((u, i) => 
                  i === index ? { 
                    ...u, 
                    status: 'success', 
                    key: response.key || response.user_id,
                    invitation_link: invitationUrl
                  } : u
                )
              );
            }
          } catch (inviteError) {
            console.error('Error creating invitation:', inviteError);
            setParsedData(current => 
              current.map((u, i) => 
                i === index ? { 
                  ...u, 
                  status: 'success', 
                  key: response.key || response.user_id,
                  error: 'User created but failed to generate invitation link'
                } : u
              )
            );
          }
        } else {
          console.log('Error case triggered');
          const errorMessage = response?.error || 'Failed to create user';
          console.log('Error message:', errorMessage);
          setParsedData(current => 
            current.map((u, i) => 
              i === index ? { ...u, status: 'failed', error: errorMessage } : u
          )
        );
        }
      } catch (error) {
        console.error('Caught error:', error);
        const errorMessage = error?.response?.data?.error || 
                           error?.message || 
                           String(error);
        setParsedData(current => 
          current.map((u, i) => 
            i === index ? { ...u, status: 'failed', error: errorMessage } : u
          )
        );
      }
    }

    setIsProcessing(false);
  };

  const downloadResults = () => {
    const results = parsedData.map(user => ({
      user_email: user.user_email,
      user_role: user.user_role,
      status: user.status,
      key: user.key || '',
      invitation_link: user.invitation_link || '',
      error: user.error || ''
    }));
    
    const csv = Papa.unparse(results);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bulk_users_results.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
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
      title: "Teams",
      dataIndex: "teams",
      key: "teams",
    },
    {
      title: "Budget",
      dataIndex: "max_budget",
      key: "max_budget",
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
            <div>
              <div className="flex items-center">
                <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                <span className="text-green-500">Success</span>
              </div>
              {record.invitation_link && (
                <div className="mt-1">
                  <div className="flex items-center">
                    <span className="text-xs text-gray-500 truncate max-w-[150px]">
                      {record.invitation_link}
                    </span>
                    <CopyToClipboard
                      text={record.invitation_link}
                      onCopy={() => message.success("Invitation link copied!")}
                    >
                      <button className="ml-1 text-blue-500 text-xs hover:text-blue-700">
                        Copy
                      </button>
                    </CopyToClipboard>
                  </div>
                </div>
              )}
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
                  <li>After creation, download the results file containing the API keys for each user</li>
                </ol>
                
                <div className="bg-gray-50 p-4 rounded-md border border-gray-200 mb-4">
                  <h4 className="font-medium mb-2">Template Column Names</h4>
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
                        <p className="text-sm text-gray-600">User's role (one of: "proxy_admin", "proxy_admin_view_only", "internal_user", "internal_user_view_only")</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">teams</p>
                        <p className="text-sm text-gray-600">Comma-separated team IDs (e.g., "team-1,team-2")</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">max_budget</p>
                        <p className="text-sm text-gray-600">Maximum budget as a number (e.g., "100")</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">budget_duration</p>
                        <p className="text-sm text-gray-600">Budget reset period (e.g., "30d", "1mo")</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="w-3 h-3 rounded-full bg-gray-300 mt-1.5 mr-2 flex-shrink-0"></div>
                      <div>
                        <p className="font-medium">models</p>
                        <p className="text-sm text-gray-600">Comma-separated allowed models (e.g., "gpt-3.5-turbo,gpt-4")</p>
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
                    {parsedData.some(user => user.status === 'success') && (
                      <Button2
                        onClick={downloadResults}
                        variant="secondary"
                        className="mr-3"
                      >
                        Download Results
                      </Button2>
                    )}
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
                  {parsedData.some(user => user.status === 'success') && (
                    <Button2
                      onClick={downloadResults}
                      variant="secondary"
                      className="mr-3"
                    >
                      Download Results
                    </Button2>
                  )}
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