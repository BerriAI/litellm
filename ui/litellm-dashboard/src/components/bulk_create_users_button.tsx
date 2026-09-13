import React, { useState, useEffect } from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, FileText, FileWarning, Trash2, TriangleAlert, Upload } from "lucide-react";
import { userCreateCall, invitationCreateCall, getProxyUISettings } from "./networking";
import Papa from "papaparse";
import { CheckCircleIcon, XCircleIcon, ExclamationIcon } from "@heroicons/react/outline";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { toast } from "@/lib/toast";

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

const PREVIEW_PAGE_SIZE = 5;

// Define an interface for the UI settings
interface UISettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
}

const BulkCreateUsersButton: React.FC<BulkCreateUsersProps> = ({
  accessToken,
  teams,
  possibleUIRoles,
  onUsersCreated,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [parsedData, setParsedData] = useState<UserData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [csvStructureError, setCsvStructureError] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("http://localhost:4000");
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);
  const csvInputId = React.useId();

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

  const handleFileUpload = (file: File) => {
    // Reset all error states
    setParseError(null);
    setCsvStructureError(null);
    setFileError(null);

    // Set the selected file - always show the file even if it's invalid
    setSelectedFile(file);

    // Check file type
    if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
      setFileError(`Invalid file type: ${file.name}. Please upload a CSV file (.csv extension).`);
      toast.fromError("Invalid file type. Please upload a CSV file.");
      return;
    }

    // Check file size (limit to 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setFileError(
        `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Please upload a CSV file smaller than 5MB.`,
      );
      return;
    }

    Papa.parse(file, {
      complete: (results) => {
        // Check if file is empty
        if (!results.data || results.data.length === 0) {
          setCsvStructureError("The CSV file appears to be empty. Please upload a file with data.");
          setParsedData([]);
          return;
        }

        // Check if there's only header row
        if (results.data.length === 1) {
          setCsvStructureError(
            "The CSV file only contains headers but no user data. Please add user data to your CSV.",
          );
          setParsedData([]);
          return;
        }

        const headers = results.data[0] as string[];

        // Check if headers exist
        if (headers.length === 0 || (headers.length === 1 && headers[0] === "")) {
          setCsvStructureError(
            "The CSV file doesn't contain any column headers. Please make sure your CSV has headers.",
          );
          setParsedData([]);
          return;
        }

        const requiredColumns = ["user_email", "user_role"];

        // Check if all required columns are present
        const missingColumns = requiredColumns.filter((col) => !headers.includes(col));
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
              // Skip empty rows
              if (row.length === 0 || (row.length === 1 && row[0] === "")) {
                return null;
              }

              // Check if row has enough columns
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
                budget_duration: row[headers.indexOf("budget_duration")]?.trim(),
                models: row[headers.indexOf("models")]?.trim(),
                rowNumber: index + 2,
                isValid: true,
                error: "",
              };

              // Validate the row
              const errors: string[] = [];

              // Email validation
              if (!user.user_email) {
                errors.push("Email is required");
              } else if (!user.user_email.includes("@") || !user.user_email.includes(".")) {
                errors.push("Invalid email format (must contain @ and domain)");
              }

              // Role validation
              if (!user.user_role) {
                errors.push("Role is required");
              } else {
                // Validate user role
                const validRoles = ["proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"];
                if (!validRoles.includes(user.user_role)) {
                  errors.push(`Invalid role "${user.user_role}". Must be one of: ${validRoles.join(", ")}`);
                }
              }

              // Budget validation
              if (user.max_budget && user.max_budget.toString().trim() !== "") {
                if (isNaN(parseFloat(user.max_budget.toString()))) {
                  errors.push(`Max budget "${user.max_budget}" must be a number`);
                } else if (parseFloat(user.max_budget.toString()) <= 0) {
                  errors.push("Max budget must be greater than 0");
                }
              }

              // Budget duration validation
              if (user.budget_duration && !user.budget_duration.match(/^\d+[dhmwy]$|^\d+mo$/)) {
                errors.push(
                  `Invalid budget duration format "${user.budget_duration}". Use format like "30d", "1mo", "2w", "6h"`,
                );
              }

              // Teams validation
              if (user.teams && typeof user.teams === "string") {
                // Check if teams exist (if teams data is available)
                if (teams && teams.length > 0) {
                  const teamIds = teams.map((t) => t.team_id);
                  const userTeams = user.teams.split(",").map((t) => t.trim());
                  const invalidTeams = userTeams.filter((t) => !teamIds.includes(t));
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
            .filter(Boolean) as UserData[]; // Filter out null values (empty rows)

          const validData = userData.filter((user) => user.isValid);
          setParsedData(userData);

          if (userData.length === 0) {
            setCsvStructureError("No valid data rows found in the CSV file. Please check your file format.");
          } else if (validData.length === 0) {
            setParseError("No valid users found in the CSV. Please check the errors below and fix your CSV file.");
          } else if (validData.length < userData.length) {
            setParseError(
              `Found ${userData.length - validData.length} row(s) with errors out of ${userData.length} total rows. Please correct them before proceeding.`,
            );
          } else {
            toast.success(`Successfully parsed ${validData.length} users`);
          }
        } catch (error: unknown) {
          const errorMessage = error instanceof Error ? error.message : "Unknown error";
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
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setParsedData([]);
    setParseError(null);
    setCsvStructureError(null);
    setFileError(null);
  };

  const resetParsedData = () => {
    setParsedData([]);
    setParseError(null);
    setPageIndex(0);
  };

  const handleBulkCreate = async () => {
    setIsProcessing(true);
    const updatedData = parsedData.map((user) => ({ ...user, status: "pending" }));
    setParsedData(updatedData);

    let anySuccessful = false;

    for (let index = 0; index < updatedData.length; index++) {
      const user = updatedData[index];
      try {
        // Create a clean user object with only non-empty values
        const cleanUser: Partial<UserData> = {
          user_email: user.user_email,
          user_role: user.user_role,
        };

        // Only add optional fields if they have values
        if (user.teams && typeof user.teams === "string" && user.teams.trim() !== "") {
          cleanUser.teams = user.teams
            .split(",")
            .map((team) => team.trim())
            .filter(Boolean);
          // Only include teams if there's at least one valid team
          if (cleanUser.teams.length === 0) {
            delete cleanUser.teams;
          }
        }

        // Only add models if provided and non-empty
        if (user.models && typeof user.models === "string" && user.models.trim() !== "") {
          cleanUser.models = user.models
            .split(",")
            .map((model) => model.trim())
            .filter(Boolean);
          // Only include models if there's at least one valid model
          if (cleanUser.models.length === 0) {
            delete cleanUser.models;
          }
        }

        // Only add max_budget if it's a valid number
        if (user.max_budget && user.max_budget.toString().trim() !== "") {
          const budgetValue = parseFloat(user.max_budget.toString());
          if (!isNaN(budgetValue) && budgetValue > 0) {
            cleanUser.max_budget = budgetValue;
          }
        }

        // Only add budget_duration if provided and non-empty
        if (user.budget_duration && user.budget_duration.trim() !== "") {
          cleanUser.budget_duration = user.budget_duration.trim();
        }

        // Only add metadata if provided and non-empty
        if (user.metadata && typeof user.metadata === "string" && user.metadata.trim() !== "") {
          cleanUser.metadata = user.metadata.trim();
        }

        const response = await userCreateCall(accessToken, null, cleanUser);

        // Check if response has key or user_id, indicating success
        if (response && (response.key || response.user_id)) {
          anySuccessful = true;
          const user_id = response.data?.user_id || response.user_id;

          // Create invitation link for the user
          try {
            if (!uiSettings?.SSO_ENABLED) {
              // Regular invitation flow
              const invitationData = await invitationCreateCall(accessToken, user_id);
              const invitationUrl = new URL(`/ui/onboarding?invitation_id=${invitationData.id}`, baseUrl).toString();

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
              // SSO flow - just use the base URL
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
                      error: "User created but failed to generate invitation link",
                    }
                  : u,
              ),
            );
          }
        } else {
          const errorMessage = response?.error || "Failed to create user";
          setParsedData((current) =>
            current.map((u, i) => (i === index ? { ...u, status: "failed", error: errorMessage } : u)),
          );
        }
      } catch (error) {
        console.error("Caught error:", error);
        const errorMessage = (error as any)?.response?.data?.error || (error as Error)?.message || String(error);
        setParsedData((current) =>
          current.map((u, i) => (i === index ? { ...u, status: "failed", error: errorMessage } : u)),
        );
      }
    }

    setIsProcessing(false);

    // Call the callback if any users were successfully created
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

  const renderStatusCell = (record: UserData) => {
    if (!record.isValid) {
      return (
        <div>
          <div className="flex items-center">
            <XCircleIcon className="h-5 w-5 text-destructive mr-2" />
            <span className="text-destructive">Invalid</span>
          </div>
          {record.error && <span className="text-sm text-destructive ml-7">{record.error}</span>}
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
            <CheckCircleIcon className="h-5 w-5 text-success mr-2" />
            <span className="text-success">Success</span>
          </div>
          {record.invitation_link && (
            <div className="mt-1">
              <div className="flex items-center">
                <span className="text-xs text-muted-foreground truncate max-w-[150px]">{record.invitation_link}</span>
                <CopyToClipboard text={record.invitation_link} onCopy={() => toast.success("Invitation link copied!")}>
                  <button className="ml-1 text-info text-xs hover:text-info/80">Copy</button>
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
          <XCircleIcon className="h-5 w-5 text-destructive mr-2" />
          <span className="text-destructive">Failed</span>
        </div>
        {record.error && <span className="text-sm text-destructive ml-7">{JSON.stringify(record.error)}</span>}
      </div>
    );
  };

  const pageCount = Math.max(1, Math.ceil(parsedData.length / PREVIEW_PAGE_SIZE));
  const currentPage = Math.min(pageIndex, pageCount - 1);
  const visibleRows = parsedData.slice(currentPage * PREVIEW_PAGE_SIZE, (currentPage + 1) * PREVIEW_PAGE_SIZE);

  return (
    <>
      <Button className="mb-0" onClick={() => setIsModalVisible(true)}>
        + Bulk Invite Users
      </Button>

      <Dialog open={isModalVisible} onOpenChange={(open) => !open && setIsModalVisible(false)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>Bulk Invite Users</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col">
            {/* Step indicator */}
            {parsedData.length === 0 ? (
              <div className="mb-6">
                <div className="flex items-center mb-4">
                  <div className="w-8 h-8 rounded-full bg-info text-info-foreground flex items-center justify-center mr-3">
                    1
                  </div>
                  <h3 className="text-lg font-medium">Download and fill the template</h3>
                </div>

                <div className="ml-11 mb-6">
                  <p className="mb-4">Add multiple users at once by following these steps:</p>
                  <ol className="list-decimal list-inside space-y-2 ml-2 mb-4">
                    <li>Download our CSV template</li>
                    <li>Add your users&apos; information to the spreadsheet</li>
                    <li>Save the file and upload it here</li>
                    <li>After creation, download the results file containing the Virtual Keys for each user</li>
                  </ol>

                  <div className="bg-muted p-4 rounded-md border border-border mb-4">
                    <h4 className="font-medium mb-2">Template Column Names</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-destructive mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">user_email</p>
                          <p className="text-sm text-muted-foreground">User&apos;s email address (required)</p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-destructive mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">user_role</p>
                          <p className="text-sm text-muted-foreground">
                            User&apos;s role (one of: &quot;proxy_admin&quot;, &quot;proxy_admin_viewer&quot;,
                            &quot;internal_user&quot;, &quot;internal_user_viewer&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-border mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">teams</p>
                          <p className="text-sm text-muted-foreground">
                            Comma-separated team IDs (e.g., &quot;team-1,team-2&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-border mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">max_budget</p>
                          <p className="text-sm text-muted-foreground">
                            Maximum budget as a number (e.g., &quot;100&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-border mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">budget_duration</p>
                          <p className="text-sm text-muted-foreground">
                            Budget reset period (e.g., &quot;30d&quot;, &quot;1mo&quot;)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="w-3 h-3 rounded-full bg-border mt-1.5 mr-2 shrink-0"></div>
                        <div>
                          <p className="font-medium">models</p>
                          <p className="text-sm text-muted-foreground">
                            Comma-separated allowed models (e.g., &quot;gpt-3.5-turbo,gpt-4&quot;)
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <Button size="lg" className="w-full md:w-auto">
                    <Download className="size-4" />
                    Download CSV Template
                  </Button>
                </div>

                <div className="flex items-center mb-4">
                  <div className="w-8 h-8 rounded-full bg-info text-info-foreground flex items-center justify-center mr-3">
                    2
                  </div>
                  <h3 className="text-lg font-medium">Upload your completed CSV</h3>
                </div>

                <div className="ml-11">
                  {selectedFile ? (
                    <div
                      className={`mb-4 p-4 rounded-md border ${fileError ? "bg-destructive/10 border-destructive/20" : "bg-info/10 border-info/20"}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center min-w-0">
                          {fileError ? (
                            <FileWarning className="size-5 shrink-0 text-destructive mr-3" />
                          ) : (
                            <FileText className="size-5 shrink-0 text-info mr-3" />
                          )}
                          <div className="min-w-0">
                            <strong className={`break-words ${fileError ? "text-destructive" : "text-info"}`}>
                              {selectedFile.name}
                            </strong>
                            <span className={`block text-xs ${fileError ? "text-destructive" : "text-info"}`}>
                              {(selectedFile.size / 1024).toFixed(1)} KB • {new Date().toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button variant="outline" size="sm" onClick={removeSelectedFile} className="flex items-center">
                          <Trash2 className="size-4" />
                          Remove
                        </Button>
                      </div>

                      {fileError ? (
                        <div className="mt-3 text-destructive text-sm flex items-start">
                          <TriangleAlert className="size-3.5 shrink-0 mr-2 mt-0.5" />
                          <span className="min-w-0 break-words">{fileError}</span>
                        </div>
                      ) : (
                        !csvStructureError && (
                          <div className="mt-3 flex items-center">
                            <div className="w-full bg-border rounded-full h-1.5">
                              <div className="bg-info h-1.5 rounded-full w-full animate-pulse"></div>
                            </div>
                            <span className="ml-2 text-xs text-info">Processing...</span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <label
                      htmlFor={csvInputId}
                      className="block"
                      onDragOver={handleDragOver}
                      onDragLeave={() => setIsDraggingOver(false)}
                      onDrop={handleDrop}
                    >
                      <div
                        className={`border-2 border-dashed ${isDraggingOver ? "border-info" : "border-border"} rounded-lg p-8 text-center hover:border-info focus-within:border-info transition-colors cursor-pointer`}
                      >
                        <input
                          id={csvInputId}
                          type="file"
                          accept=".csv"
                          className="sr-only"
                          onChange={handleFileInputChange}
                        />
                        <Upload className="size-[30px] text-muted-foreground mb-2" />
                        <p className="mb-1">Drag and drop your CSV file here</p>
                        <p className="text-sm text-muted-foreground mb-3">or</p>
                        <span className={buttonVariants({ variant: "outline", size: "sm" })}>Browse files</span>
                        <p className="text-xs text-muted-foreground mt-4">Only CSV files (.csv) are supported</p>
                      </div>
                    </label>
                  )}

                  {csvStructureError && (
                    <div className="mb-4 p-4 bg-warning/10 border border-warning/20 rounded-md">
                      <div className="flex items-start">
                        <ExclamationIcon className="h-5 w-5 shrink-0 text-warning mr-2 mt-0.5" />
                        <div className="min-w-0">
                          <strong className="text-warning">CSV Structure Error</strong>
                          <p className="text-warning mt-1 mb-0 break-words">{csvStructureError}</p>
                          <p className="text-warning mt-2 mb-0">
                            Please download our template and ensure your CSV follows the required format.
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
                  <div className="w-8 h-8 rounded-full bg-info text-info-foreground flex items-center justify-center mr-3">
                    3
                  </div>
                  <h3 className="text-lg font-medium">
                    {parsedData.some((user) => user.status === "success" || user.status === "failed")
                      ? "User Creation Results"
                      : "Review and create users"}
                  </h3>
                </div>

                {parseError && (
                  <div className="ml-11 mb-4 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
                    <div className="flex items-start">
                      <TriangleAlert className="size-4 shrink-0 text-destructive mr-2 mt-1" />
                      <div className="min-w-0">
                        <p className="text-destructive font-medium break-words">{parseError}</p>
                        {parsedData.some((user) => !user.isValid) && (
                          <ul className="mt-2 list-disc list-inside text-destructive text-sm">
                            <li>Check the table below for specific errors in each row</li>
                            <li>
                              Common issues include invalid email formats, missing required fields, or incorrect role
                              values
                            </li>
                            <li>Fix these issues in your CSV file and upload again</li>
                          </ul>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div className="ml-11">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center">
                      {parsedData.some((user) => user.status === "success" || user.status === "failed") ? (
                        <div className="flex items-center">
                          <p className="text-lg font-medium mr-3">Creation Summary</p>
                          <p className="text-sm bg-success/15 text-success px-2 py-1 rounded-sm mr-2">
                            {parsedData.filter((d) => d.status === "success").length} Successful
                          </p>
                          {parsedData.some((d) => d.status === "failed") && (
                            <p className="text-sm bg-destructive/15 text-destructive px-2 py-1 rounded-sm">
                              {parsedData.filter((d) => d.status === "failed").length} Failed
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center">
                          <p className="text-lg font-medium mr-3">User Preview</p>
                          <p className="text-sm bg-info/15 text-info px-2 py-1 rounded-sm">
                            {parsedData.filter((d) => d.isValid).length} of {parsedData.length} users valid
                          </p>
                        </div>
                      )}
                    </div>

                    {!parsedData.some((user) => user.status === "success" || user.status === "failed") && (
                      <div className="flex space-x-3">
                        <Button variant="outline" onClick={resetParsedData}>
                          Back
                        </Button>
                        <Button
                          onClick={handleBulkCreate}
                          disabled={parsedData.filter((d) => d.isValid).length === 0 || isProcessing}
                        >
                          {isProcessing ? "Creating..." : `Create ${parsedData.filter((d) => d.isValid).length} Users`}
                        </Button>
                      </div>
                    )}
                  </div>

                  {parsedData.some((user) => user.status === "success") && (
                    <div className="mb-4 p-4 bg-info/10 border border-info/20 rounded-md">
                      <div className="flex items-start">
                        <div className="mr-3 mt-1">
                          <CheckCircleIcon className="h-5 w-5 text-info" />
                        </div>
                        <div>
                          <p className="font-medium text-info">User creation complete</p>
                          <p className="block text-sm text-info mt-1">
                            <span className="font-medium">Next step:</span> Download the credentials file containing
                            Virtual Keys and invitation links. Users will need these Virtual Keys to make LLM requests
                            through LiteLLM.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="max-h-[300px] overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-20">Row</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Teams</TableHead>
                          <TableHead>Budget</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {visibleRows.map((record) => (
                          <TableRow key={record.rowNumber} className={!record.isValid ? "bg-destructive/10" : ""}>
                            <TableCell>{record.rowNumber}</TableCell>
                            <TableCell className="whitespace-normal break-words">{record.user_email}</TableCell>
                            <TableCell className="whitespace-normal break-words">{record.user_role}</TableCell>
                            <TableCell className="whitespace-normal break-words">{record.teams}</TableCell>
                            <TableCell>{record.max_budget}</TableCell>
                            <TableCell className="whitespace-normal break-words">{renderStatusCell(record)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  {pageCount > 1 && (
                    <div className="flex items-center justify-end gap-3 mt-2">
                      <span className="text-sm text-muted-foreground">
                        Page {currentPage + 1} of {pageCount}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setPageIndex(currentPage - 1)}
                        disabled={currentPage === 0}
                      >
                        Previous
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setPageIndex(currentPage + 1)}
                        disabled={currentPage >= pageCount - 1}
                      >
                        Next
                      </Button>
                    </div>
                  )}

                  {!parsedData.some((user) => user.status === "success" || user.status === "failed") && (
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" onClick={resetParsedData} className="mr-3">
                        Back
                      </Button>
                      <Button
                        onClick={handleBulkCreate}
                        disabled={parsedData.filter((d) => d.isValid).length === 0 || isProcessing}
                      >
                        {isProcessing ? "Creating..." : `Create ${parsedData.filter((d) => d.isValid).length} Users`}
                      </Button>
                    </div>
                  )}

                  {parsedData.some((user) => user.status === "success" || user.status === "failed") && (
                    <div className="flex justify-end mt-4">
                      <Button variant="outline" onClick={resetParsedData} className="mr-3">
                        Start New Bulk Import
                      </Button>
                      <Button onClick={downloadResults}>
                        <Download className="size-4" />
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
