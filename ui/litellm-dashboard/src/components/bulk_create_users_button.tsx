import React, { useEffect, useRef, useState } from "react";
import Papa from "papaparse";
import { CopyToClipboard } from "react-copy-to-clipboard";
import {
  AlertTriangle as WarningOutlined,
  CheckCircle2 as CheckCircleIcon,
  Download as DownloadOutlined,
  FileText as FileTextOutlined,
  FileWarning as FileExclamationOutlined,
  Trash2 as DeleteOutlined,
  Upload as UploadOutlined,
  XCircle as XCircleIcon,
  AlertTriangle as ExclamationIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import {
  userCreateCall,
  invitationCreateCall,
  getProxyUISettings,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

interface BulkCreateUsersProps {
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onUsersCreated?: () => void;
}

interface UserData {
  user_email: string;
  user_role: string;
  teams?: string | string[];
  metadata?: string;
  max_budget?: string | number;
  budget_duration?: string;
  models?: string | string[];
  status?: string;
  error?: string;
  rowNumber?: number;
  isValid?: boolean;
  key?: string;
  invitation_link?: string;
}

interface UISettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
}

const PAGE_SIZE = 5;

const BulkCreateUsersButton: React.FC<BulkCreateUsersProps> = ({
  accessToken,
  teams,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  possibleUIRoles,
  onUsersCreated,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [parsedData, setParsedData] = useState<UserData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [csvStructureError, setCsvStructureError] = useState<string | null>(
    null,
  );
  const [fileError, setFileError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("http://localhost:4000");
  const [page, setPage] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchUISettings = async () => {
      try {
        const uiSettingsResponse = await getProxyUISettings(accessToken);
        setUISettings(uiSettingsResponse);
      } catch (error) {
        console.error("Error fetching UI settings:", error);
      }
    };

    fetchUISettings();

    const base = new URL("/", window.location.href);
    setBaseUrl(base.toString());
  }, [accessToken]);

  const downloadTemplate = () => {
    const template = [
      [
        "user_email",
        "user_role",
        "teams",
        "max_budget",
        "budget_duration",
        "models",
      ],
      [
        "user@example.com",
        "internal_user",
        "team-id-1,team-id-2",
        "100",
        "30d",
        "gpt-3.5-turbo,gpt-4",
      ],
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
    setCsvStructureError(null);
    setFileError(null);

    setSelectedFile(file);

    if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
      setFileError(
        `Invalid file type: ${file.name}. Please upload a CSV file (.csv extension).`,
      );
      NotificationsManager.fromBackend(
        "Invalid file type. Please upload a CSV file.",
      );
      return false;
    }

    if (file.size > 5 * 1024 * 1024) {
      setFileError(
        `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Please upload a CSV file smaller than 5MB.`,
      );
      return false;
    }

    Papa.parse(file, {
      complete: (results) => {
        if (!results.data || results.data.length === 0) {
          setCsvStructureError(
            "The CSV file appears to be empty. Please upload a file with data.",
          );
          setParsedData([]);
          return;
        }

        if (results.data.length === 1) {
          setCsvStructureError(
            "The CSV file only contains headers but no user data. Please add user data to your CSV.",
          );
          setParsedData([]);
          return;
        }

        const headers = results.data[0] as string[];

        if (headers.length === 0 || (headers.length === 1 && headers[0] === "")) {
          setCsvStructureError(
            "The CSV file doesn't contain any column headers. Please make sure your CSV has headers.",
          );
          setParsedData([]);
          return;
        }

        const requiredColumns = ["user_email", "user_role"];

        const missingColumns = requiredColumns.filter(
          (col) => !headers.includes(col),
        );
        if (missingColumns.length > 0) {
          setCsvStructureError(
            `Your CSV is missing these required columns: ${missingColumns.join(", ")}. Please add these columns to your CSV file.`,
          );
          setParsedData([]);
          return;
        }

        try {
          const userData = results.data
            .slice(1)
            .map((row: any, index: number) => {
              if (row.length === 0 || (row.length === 1 && row[0] === "")) {
                return null;
              }

              if (row.length < headers.length) {
                return {
                  rowNumber: index + 2,
                  isValid: false,
                  error: `Row ${index + 2} has fewer columns than the header row. Please ensure all data is properly formatted.`,
                  user_email: "",
                  user_role: "",
                } as UserData;
              }

              const user: UserData = {
                user_email: row[headers.indexOf("user_email")]?.trim() || "",
                user_role: row[headers.indexOf("user_role")]?.trim() || "",
                teams: row[headers.indexOf("teams")]?.trim(),
                max_budget: row[headers.indexOf("max_budget")]?.trim(),
                budget_duration:
                  row[headers.indexOf("budget_duration")]?.trim(),
                models: row[headers.indexOf("models")]?.trim(),
                rowNumber: index + 2,
                isValid: true,
                error: "",
              };

              const errors: string[] = [];

              if (!user.user_email) {
                errors.push("Email is required");
              } else if (
                !user.user_email.includes("@") ||
                !user.user_email.includes(".")
              ) {
                errors.push("Invalid email format (must contain @ and domain)");
              }

              if (!user.user_role) {
                errors.push("Role is required");
              } else {
                const validRoles = [
                  "proxy_admin",
                  "proxy_admin_viewer",
                  "internal_user",
                  "internal_user_viewer",
                ];
                if (!validRoles.includes(user.user_role)) {
                  errors.push(
                    `Invalid role "${user.user_role}". Must be one of: ${validRoles.join(", ")}`,
                  );
                }
              }

              if (user.max_budget && user.max_budget.toString().trim() !== "") {
                if (isNaN(parseFloat(user.max_budget.toString()))) {
                  errors.push(
                    `Max budget "${user.max_budget}" must be a number`,
                  );
                } else if (parseFloat(user.max_budget.toString()) <= 0) {
                  errors.push("Max budget must be greater than 0");
                }
              }

              if (
                user.budget_duration &&
                !user.budget_duration.match(/^\d+[dhmwy]$|^\d+mo$/)
              ) {
                errors.push(
                  `Invalid budget duration format "${user.budget_duration}". Use format like "30d", "1mo", "2w", "6h"`,
                );
              }

              if (user.teams && typeof user.teams === "string") {
                if (teams && teams.length > 0) {
                  const teamIds = teams.map((t) => t.team_id);
                  const userTeams = user.teams
                    .split(",")
                    .map((t) => t.trim());
                  const invalidTeams = userTeams.filter(
                    (t) => !teamIds.includes(t),
                  );
                  if (invalidTeams.length > 0) {
                    errors.push(`Unknown team(s): ${invalidTeams.join(", ")}`);
                  }
                }
              }

              if (errors.length > 0) {
                user.isValid = false;
                user.error = errors.join(", ");
              }

              return user;
            })
            .filter(Boolean) as UserData[];

          const validData = userData.filter((user) => user.isValid);
          setParsedData(userData);
          setPage(0);

          if (userData.length === 0) {
            setCsvStructureError(
              "No valid data rows found in the CSV file. Please check your file format.",
            );
          } else if (validData.length === 0) {
            setParseError(
              "No valid users found in the CSV. Please check the errors below and fix your CSV file.",
            );
          } else if (validData.length < userData.length) {
            setParseError(
              `Found ${userData.length - validData.length} row(s) with errors out of ${userData.length} total rows. Please correct them before proceeding.`,
            );
          } else {
            NotificationsManager.success(
              `Successfully parsed ${validData.length} users`,
            );
          }
        } catch (error: unknown) {
          const errorMessage =
            error instanceof Error ? error.message : "Unknown error";
          setParseError(`Error parsing CSV: ${errorMessage}`);
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

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setParsedData([]);
    setParseError(null);
    setCsvStructureError(null);
    setFileError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleBulkCreate = async () => {
    setIsProcessing(true);
    const updatedData = parsedData.map((user) => ({
      ...user,
      status: "pending",
    }));
    setParsedData(updatedData);

    let anySuccessful = false;

    for (let index = 0; index < updatedData.length; index++) {
      const user = updatedData[index];
      try {
        const cleanUser: Partial<UserData> = {
          user_email: user.user_email,
          user_role: user.user_role,
        };

        if (
          user.teams &&
          typeof user.teams === "string" &&
          user.teams.trim() !== ""
        ) {
          cleanUser.teams = user.teams
            .split(",")
            .map((team) => team.trim())
            .filter(Boolean);
          if (cleanUser.teams.length === 0) {
            delete cleanUser.teams;
          }
        }

        if (
          user.models &&
          typeof user.models === "string" &&
          user.models.trim() !== ""
        ) {
          cleanUser.models = user.models
            .split(",")
            .map((model) => model.trim())
            .filter(Boolean);
          if (cleanUser.models.length === 0) {
            delete cleanUser.models;
          }
        }

        if (user.max_budget && user.max_budget.toString().trim() !== "") {
          const budgetValue = parseFloat(user.max_budget.toString());
          if (!isNaN(budgetValue) && budgetValue > 0) {
            cleanUser.max_budget = budgetValue;
          }
        }

        if (user.budget_duration && user.budget_duration.trim() !== "") {
          cleanUser.budget_duration = user.budget_duration.trim();
        }

        if (
          user.metadata &&
          typeof user.metadata === "string" &&
          user.metadata.trim() !== ""
        ) {
          cleanUser.metadata = user.metadata.trim();
        }

        console.log("Sending user data:", cleanUser);
        const response = await userCreateCall(accessToken, null, cleanUser);
        console.log("Full response:", response);

        if (response && (response.key || response.user_id)) {
          anySuccessful = true;
          console.log("Success case triggered");
          const user_id = response.data?.user_id || response.user_id;

          try {
            if (!uiSettings?.SSO_ENABLED) {
              const invitationData = await invitationCreateCall(
                accessToken,
                user_id,
              );
              const invitationUrl = new URL(
                `/ui?invitation_id=${invitationData.id}`,
                baseUrl,
              ).toString();

              setParsedData((current) =>
                current.map((u, i) =>
                  i === index
                    ? {
                        ...u,
                        status: "success",
                        key: response.key || response.user_id,
                        invitation_link: invitationUrl,
                      }
                    : u,
                ),
              );
            } else {
              const invitationUrl = new URL("/ui", baseUrl).toString();

              setParsedData((current) =>
                current.map((u, i) =>
                  i === index
                    ? {
                        ...u,
                        status: "success",
                        key: response.key || response.user_id,
                        invitation_link: invitationUrl,
                      }
                    : u,
                ),
              );
            }
          } catch (inviteError) {
            console.error("Error creating invitation:", inviteError);
            setParsedData((current) =>
              current.map((u, i) =>
                i === index
                  ? {
                      ...u,
                      status: "success",
                      key: response.key || response.user_id,
                      error:
                        "User created but failed to generate invitation link",
                    }
                  : u,
              ),
            );
          }
        } else {
          console.log("Error case triggered");
          const errorMessage = response?.error || "Failed to create user";
          console.log("Error message:", errorMessage);
          setParsedData((current) =>
            current.map((u, i) =>
              i === index
                ? { ...u, status: "failed", error: errorMessage }
                : u,
            ),
          );
        }
      } catch (error) {
        console.error("Caught error:", error);
        const errorMessage =
          (error as any)?.response?.data?.error ||
          (error as Error)?.message ||
          String(error);
        setParsedData((current) =>
          current.map((u, i) =>
            i === index ? { ...u, status: "failed", error: errorMessage } : u,
          ),
        );
      }
    }

    setIsProcessing(false);

    if (anySuccessful && onUsersCreated) {
      onUsersCreated();
    }
  };

  const downloadResults = () => {
    const results = parsedData.map((user) => ({
      user_email: user.user_email,
      user_role: user.user_role,
      status: user.status,
      key: user.key || "",
      invitation_link: user.invitation_link || "",
      error: user.error || "",
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

  const totalPages = Math.max(1, Math.ceil(parsedData.length / PAGE_SIZE));
  const pageStart = page * PAGE_SIZE;
  const pageRows = parsedData.slice(pageStart, pageStart + PAGE_SIZE);

  const renderStatusCell = (record: UserData) => {
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
    if (!record.status || record.status === "pending") {
      return <span className="text-muted-foreground">Pending</span>;
    }
    if (record.status === "success") {
      return (
        <div>
          <div className="flex items-center">
            <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
            <span className="text-green-500">Success</span>
          </div>
          {record.invitation_link && (
            <div className="mt-1">
              <div className="flex items-center">
                <span className="text-xs text-muted-foreground truncate max-w-[150px]">
                  {record.invitation_link}
                </span>
                <CopyToClipboard
                  text={record.invitation_link}
                  onCopy={() =>
                    NotificationsManager.success("Invitation link copied!")
                  }
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
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <>
      <Button onClick={() => setIsModalVisible(true)}>
        + Bulk Invite Users
      </Button>

      <Dialog
        open={isModalVisible}
        onOpenChange={(o) => setIsModalVisible(o)}
      >
        <DialogContent className="max-w-[800px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Bulk Invite Users</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col">
            {parsedData.length === 0 ? (
              <div className="mb-6">
                <div className="flex items-center mb-4">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">
                    1
                  </div>
                  <h3 className="text-lg font-medium">
                    Download and fill the template
                  </h3>
                </div>

                <div className="ml-11 mb-6">
                  <p className="mb-4">
                    Add multiple users at once by following these steps:
                  </p>
                  <ol className="list-decimal list-inside space-y-2 ml-2 mb-4">
                    <li>Download our CSV template</li>
                    <li>
                      Add your users&apos; information to the spreadsheet
                    </li>
                    <li>Save the file and upload it here</li>
                    <li>
                      After creation, download the results file containing the
                      Virtual Keys for each user
                    </li>
                  </ol>

                  <div className="bg-muted/50 p-4 rounded-md border border-border mb-4">
                    <h4 className="font-medium mb-2">Template Column Names</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">user_email</p>
                          <p className="text-sm text-muted-foreground">
                            User&apos;s email address (required)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">user_role</p>
                          <p className="text-sm text-muted-foreground">
                            User&apos;s role (one of:
                            &quot;proxy_admin&quot;,
                            &quot;proxy_admin_viewer&quot;,
                            &quot;internal_user&quot;,
                            &quot;internal_user_viewer&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-muted-foreground/50 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">teams</p>
                          <p className="text-sm text-muted-foreground">
                            Comma-separated team IDs (e.g.,
                            &quot;team-1,team-2&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-muted-foreground/50 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">max_budget</p>
                          <p className="text-sm text-muted-foreground">
                            Maximum budget as a number (e.g.,
                            &quot;100&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-muted-foreground/50 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">budget_duration</p>
                          <p className="text-sm text-muted-foreground">
                            Budget reset period (e.g., &quot;30d&quot;,
                            &quot;1mo&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-muted-foreground/50 mt-1.5 mr-2 flex-shrink-0"></div>
                        <div>
                          <p className="font-medium">models</p>
                          <p className="text-sm text-muted-foreground">
                            Comma-separated allowed models (e.g.,
                            &quot;gpt-3.5-turbo,gpt-4&quot;)
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <Button
                    size="lg"
                    className="w-full md:w-auto"
                    onClick={downloadTemplate}
                  >
                    <DownloadOutlined className="h-4 w-4 mr-1" />
                    Download CSV Template
                  </Button>
                </div>

                <div className="flex items-center mb-4">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">
                    2
                  </div>
                  <h3 className="text-lg font-medium">
                    Upload your completed CSV
                  </h3>
                </div>

                <div className="ml-11">
                  {selectedFile ? (
                    <div
                      className={`mb-4 p-4 rounded-md border ${fileError ? "bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900" : "bg-blue-50 border-blue-200 dark:bg-blue-950/30 dark:border-blue-900"}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          {fileError ? (
                            <FileExclamationOutlined className="text-red-500 text-xl mr-3" />
                          ) : (
                            <FileTextOutlined className="text-blue-500 text-xl mr-3" />
                          )}
                          <div>
                            <span
                              className={`font-semibold ${fileError ? "text-red-800 dark:text-red-300" : "text-blue-800 dark:text-blue-300"}`}
                            >
                              {selectedFile.name}
                            </span>
                            <span
                              className={`block text-xs ${fileError ? "text-red-600 dark:text-red-400" : "text-blue-600 dark:text-blue-400"}`}
                            >
                              {(selectedFile.size / 1024).toFixed(1)} KB •{" "}
                              {new Date().toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={removeSelectedFile}
                          className="flex items-center"
                        >
                          <DeleteOutlined className="h-4 w-4 mr-1" />
                          Remove
                        </Button>
                      </div>

                      {fileError ? (
                        <div className="mt-3 text-red-600 dark:text-red-400 text-sm flex items-start">
                          <WarningOutlined className="mr-2 mt-0.5" />
                          <span>{fileError}</span>
                        </div>
                      ) : (
                        !csvStructureError && (
                          <div className="mt-3 flex items-center">
                            <div className="w-full bg-muted rounded-full h-1.5">
                              <div className="bg-blue-500 h-1.5 rounded-full w-full animate-pulse"></div>
                            </div>
                            <span className="ml-2 text-xs text-blue-600 dark:text-blue-400">
                              Processing...
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            handleFileUpload(e.target.files[0]);
                          }
                        }}
                      />
                      <div
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                        className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer"
                      >
                        <UploadOutlined className="h-8 w-8 text-muted-foreground mb-2 mx-auto" />
                        <p className="mb-1">
                          Drag and drop your CSV file here
                        </p>
                        <p className="text-sm text-muted-foreground mb-3">or</p>
                        <Button size="sm" variant="outline" type="button">
                          Browse files
                        </Button>
                        <p className="text-xs text-muted-foreground mt-4">
                          Only CSV files (.csv) are supported
                        </p>
                      </div>
                    </>
                  )}

                  {csvStructureError && (
                    <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-md dark:bg-yellow-950/30 dark:border-yellow-900">
                      <div className="flex items-start">
                        <ExclamationIcon className="h-5 w-5 text-yellow-500 mr-2 mt-0.5" />
                        <div>
                          <span className="font-semibold text-yellow-800 dark:text-yellow-300">
                            CSV Structure Error
                          </span>
                          <p className="text-yellow-700 dark:text-yellow-400 mt-1 mb-0">
                            {csvStructureError}
                          </p>
                          <p className="text-yellow-700 dark:text-yellow-400 mt-2 mb-0">
                            Please download our template and ensure your CSV
                            follows the required format.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="mb-6">
                <div className="flex items-center mb-4">
                  <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center mr-3">
                    3
                  </div>
                  <h3 className="text-lg font-medium">
                    {parsedData.some(
                      (user) =>
                        user.status === "success" || user.status === "failed",
                    )
                      ? "User Creation Results"
                      : "Review and create users"}
                  </h3>
                </div>

                {parseError && (
                  <div className="ml-11 mb-4 p-4 bg-red-50 border border-red-200 rounded-md dark:bg-red-950/30 dark:border-red-900">
                    <div className="flex items-start">
                      <WarningOutlined className="text-red-500 mr-2 mt-1" />
                      <div>
                        <span className="text-red-600 dark:text-red-400 font-medium">
                          {parseError}
                        </span>
                        {parsedData.some((user) => !user.isValid) && (
                          <ul className="mt-2 list-disc list-inside text-red-600 dark:text-red-400 text-sm">
                            <li>
                              Check the table below for specific errors in each
                              row
                            </li>
                            <li>
                              Common issues include invalid email formats,
                              missing required fields, or incorrect role values
                            </li>
                            <li>
                              Fix these issues in your CSV file and upload
                              again
                            </li>
                          </ul>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div className="ml-11">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center">
                      {parsedData.some(
                        (user) =>
                          user.status === "success" ||
                          user.status === "failed",
                      ) ? (
                        <div className="flex items-center">
                          <span className="text-lg font-medium mr-3">
                            Creation Summary
                          </span>
                          <span className="text-sm bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 px-2 py-1 rounded mr-2">
                            {
                              parsedData.filter((d) => d.status === "success")
                                .length
                            }{" "}
                            Successful
                          </span>
                          {parsedData.some((d) => d.status === "failed") && (
                            <span className="text-sm bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 px-2 py-1 rounded">
                              {
                                parsedData.filter(
                                  (d) => d.status === "failed",
                                ).length
                              }{" "}
                              Failed
                            </span>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center">
                          <span className="text-lg font-medium mr-3">
                            User Preview
                          </span>
                          <span className="text-sm bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 px-2 py-1 rounded">
                            {parsedData.filter((d) => d.isValid).length} of{" "}
                            {parsedData.length} users valid
                          </span>
                        </div>
                      )}
                    </div>

                    {!parsedData.some(
                      (user) =>
                        user.status === "success" || user.status === "failed",
                    ) && (
                      <div className="flex space-x-3">
                        <Button
                          variant="outline"
                          onClick={() => {
                            setParsedData([]);
                            setParseError(null);
                          }}
                        >
                          Back
                        </Button>
                        <Button
                          onClick={handleBulkCreate}
                          disabled={
                            parsedData.filter((d) => d.isValid).length === 0 ||
                            isProcessing
                          }
                        >
                          {isProcessing
                            ? "Creating..."
                            : `Create ${parsedData.filter((d) => d.isValid).length} Users`}
                        </Button>
                      </div>
                    )}
                  </div>

                  {parsedData.some((user) => user.status === "success") && (
                    <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-md dark:bg-blue-950/30 dark:border-blue-900">
                      <div className="flex items-start">
                        <div className="mr-3 mt-1">
                          <CheckCircleIcon className="h-5 w-5 text-blue-500" />
                        </div>
                        <div>
                          <span className="font-medium text-blue-800 dark:text-blue-300">
                            User creation complete
                          </span>
                          <span className="block text-sm text-blue-700 dark:text-blue-400 mt-1">
                            <span className="font-medium">Next step:</span>{" "}
                            Download the credentials file containing Virtual
                            Keys and invitation links. Users will need these
                            Virtual Keys to make LLM requests through LiteLLM.
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="border border-border rounded-md max-h-[300px] overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[80px]">Row</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Teams</TableHead>
                          <TableHead>Budget</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {pageRows.map((record, i) => (
                          <TableRow
                            key={`${pageStart + i}`}
                            className={!record.isValid ? "bg-red-50 dark:bg-red-950/20" : undefined}
                          >
                            <TableCell>{record.rowNumber}</TableCell>
                            <TableCell>{record.user_email}</TableCell>
                            <TableCell>{record.user_role}</TableCell>
                            <TableCell>
                              {typeof record.teams === "string"
                                ? record.teams
                                : record.teams?.join(", ") ?? ""}
                            </TableCell>
                            <TableCell>
                              {record.max_budget?.toString() ?? ""}
                            </TableCell>
                            <TableCell>{renderStatusCell(record)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  {parsedData.length > PAGE_SIZE && (
                    <div className="flex justify-end items-center gap-2 mt-2 text-sm">
                      <span>
                        Page {page + 1} of {totalPages}
                      </span>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={page === 0}
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                      >
                        Previous
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={page + 1 >= totalPages}
                        onClick={() =>
                          setPage((p) => Math.min(totalPages - 1, p + 1))
                        }
                      >
                        Next
                      </Button>
                    </div>
                  )}

                  {!parsedData.some(
                    (user) =>
                      user.status === "success" || user.status === "failed",
                  ) && (
                    <div className="flex justify-end mt-4">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setParsedData([]);
                          setParseError(null);
                        }}
                        className="mr-3"
                      >
                        Back
                      </Button>
                      <Button
                        onClick={handleBulkCreate}
                        disabled={
                          parsedData.filter((d) => d.isValid).length === 0 ||
                          isProcessing
                        }
                      >
                        {isProcessing
                          ? "Creating..."
                          : `Create ${parsedData.filter((d) => d.isValid).length} Users`}
                      </Button>
                    </div>
                  )}

                  {parsedData.some(
                    (user) =>
                      user.status === "success" || user.status === "failed",
                  ) && (
                    <div className="flex justify-end mt-4">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setParsedData([]);
                          setParseError(null);
                        }}
                        className="mr-3"
                      >
                        Start New Bulk Import
                      </Button>
                      <Button onClick={downloadResults}>
                        <DownloadOutlined className="h-4 w-4 mr-1" />
                        Download User Credentials
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default BulkCreateUsersButton;
