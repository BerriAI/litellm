import React, { useState, useEffect, useRef } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Badge,
  Button as TremorButton,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Button, Modal } from "antd";
import { ArrowLeftIcon, TrashIcon, PencilIcon } from "@heroicons/react/outline";
import { getPromptInfo, getPromptVersions, PromptSpec, PromptTemplateBase, deletePromptCall } from "@/components/networking";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";
import PromptCodeSnippets from "./prompt_editor_view/PromptCodeSnippets";
import {
  extractModel,
  extractTemplateVariables,
  getBasePromptId,
  getCurrentVersion
} from "./prompt_utils";

export interface PromptInfoProps {
  promptId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
  onDelete?: () => void;
  onEdit?: (promptData: any) => void;
}

const PromptInfoView: React.FC<PromptInfoProps> = ({ promptId, onClose, accessToken, isAdmin, onDelete, onEdit }) => {
  const [promptData, setPromptData] = useState<PromptSpec | null>(null);
  const [promptTemplate, setPromptTemplate] = useState<PromptTemplateBase | null>(null);
  const [rawApiResponse, setRawApiResponse] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Environment and version state
  const [environments, setEnvironments] = useState<string[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string | null>(null);
  const [versionHistory, setVersionHistory] = useState<PromptSpec[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [loadingVersions, setLoadingVersions] = useState(false);

  // Initial fetch — no environment filter, gets default + all environments list
  const fetchPromptInfo = async (environment?: string) => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await getPromptInfo(accessToken, promptId, environment);
      setPromptData(response.prompt_spec);
      setPromptTemplate(response.raw_prompt_template);
      setRawApiResponse(response);

      // Set environments from response
      if (response.environments && response.environments.length > 0) {
        setEnvironments(response.environments);
        if (!selectedEnv) {
          setSelectedEnv(response.prompt_spec.environment || response.environments[0]);
        }
      }
      setSelectedVersion(response.prompt_spec.version || null);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to load prompt information");
      console.error("Error fetching prompt info:", error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch version history for selected environment
  const fetchVersionHistory = async (env: string) => {
    if (!accessToken) return;
    setLoadingVersions(true);
    try {
      const response = await getPromptVersions(accessToken, promptId, env);
      setVersionHistory(response.prompts || []);
    } catch {
      setVersionHistory([]);
    } finally {
      setLoadingVersions(false);
    }
  };

  const isInitialMount = useRef(true);

  useEffect(() => {
    setSelectedEnv(null);
    setEnvironments([]);
    setVersionHistory([]);
    fetchPromptInfo();
  }, [promptId, accessToken]);

  // When environment changes (user clicks tab), re-fetch — skip initial mount
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      // Still fetch version history on initial mount once selectedEnv is set
      if (selectedEnv && accessToken) {
        fetchVersionHistory(selectedEnv);
      }
      return;
    }
    if (selectedEnv && accessToken) {
      fetchPromptInfo(selectedEnv);
      fetchVersionHistory(selectedEnv);
    }
  }, [selectedEnv]);

  if (loading && !promptData) {
    return <div className="p-4">Loading...</div>;
  }

  if (!promptData) {
    return <div className="p-4">Prompt not found</div>;
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const handleDeleteClick = () => {
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!accessToken || !promptData) return;
    setIsDeleting(true);
    try {
      await deletePromptCall(accessToken, basePromptId);
      NotificationsManager.success(`Prompt "${basePromptId}" deleted successfully`);
      onDelete?.();
      onClose();
    } catch (error) {
      console.error("Error deleting prompt:", error);
      NotificationsManager.fromBackend("Failed to delete prompt");
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
  };

  const handleVersionClick = async (version: PromptSpec) => {
    if (!accessToken || !selectedEnv) return;
    // Fetch specific version's info
    const versionNum = version.version || 1;
    setSelectedVersion(versionNum);
    try {
      const versionedId = `${promptId}.v${versionNum}`;
      const response = await getPromptInfo(accessToken, versionedId, selectedEnv);
      setPromptData(response.prompt_spec);
      setPromptTemplate(response.raw_prompt_template);
      setRawApiResponse(response);
    } catch {
      NotificationsManager.fromBackend(`Failed to load version v${versionNum}`);
    }
  };

  const promptModel = promptData ? extractModel(promptData) || "gpt-4o" : "gpt-4o";
  const basePromptId = getBasePromptId(promptData);
  const currentVersion = getCurrentVersion(promptData);
  const latestVersion = versionHistory.length > 0 ? Math.max(...versionHistory.map(v => v.version || 1)) : null;
  const isViewingOldVersion = latestVersion !== null && selectedVersion !== null && selectedVersion < latestVersion;

  return (
    <div className="p-4">
      <div>
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Prompts
        </TremorButton>
        <div className="flex justify-between items-start mb-4">
          <div>
            <Title>Prompt Details</Title>
            <div className="flex items-center cursor-pointer">
              <Text className="text-gray-500 font-mono">{basePromptId}</Text>
              <Button
                type="text"
                size="small"
                icon={copiedStates["prompt-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                onClick={() => copyToClipboard(basePromptId, "prompt-id")}
                className={`left-2 z-10 transition-all duration-200 ${
                  copiedStates["prompt-id"]
                    ? "text-green-600 bg-green-50 border-green-200"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
              />
            </div>
          </div>
          <div className="flex gap-2">
            <PromptCodeSnippets
              promptId={basePromptId}
              model={promptModel}
              promptVariables={extractTemplateVariables(promptTemplate?.content)}
              accessToken={accessToken}
              version={currentVersion}
            />
            <TremorButton
              icon={PencilIcon}
              variant="primary"
              onClick={() => onEdit?.(rawApiResponse)}
              className="flex items-center"
            >
              Prompt Studio
            </TremorButton>
          {isAdmin && (
            <TremorButton
              icon={TrashIcon}
              variant="secondary"
              onClick={handleDeleteClick}
              className="flex items-center"
            >
              Delete Prompt
            </TremorButton>
          )}
          </div>
        </div>
      </div>

      {/* Environment Tabs */}
      {environments.length > 0 && (
        <div className="flex gap-2 mb-4">
          {[...environments].sort((a, b) => {
            const order: Record<string, number> = { development: 0, staging: 1, production: 2 };
            return (order[a] ?? 99) - (order[b] ?? 99);
          }).map((env) => (
            <button
              key={env}
              onClick={() => {
                setSelectedEnv(env);
                setSelectedVersion(null);
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedEnv === env
                  ? env === "production"
                    ? "bg-red-100 text-red-800 border-2 border-red-300"
                    : env === "staging"
                    ? "bg-yellow-100 text-yellow-800 border-2 border-yellow-300"
                    : "bg-green-100 text-green-800 border-2 border-green-300"
                  : "bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200"
              }`}
            >
              {env}
              {versionHistory.length > 0 && selectedEnv === env && (
                <span className="ml-1 text-xs opacity-75">
                  (v{latestVersion})
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Old version banner */}
      {isViewingOldVersion && (
        <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center justify-between">
          <Text className="text-amber-800">
            Viewing v{selectedVersion} — not the latest version (v{latestVersion})
          </Text>
          <TremorButton
            variant="light"
            size="xs"
            onClick={() => {
              const latest = versionHistory.find(v => v.version === latestVersion);
              if (latest) handleVersionClick(latest);
            }}
          >
            Go to latest
          </TremorButton>
        </div>
      )}

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">Overview</Tab>
          {promptTemplate ? <Tab key="prompt-template">Prompt Template</Tab> : <></>}
          <Tab key="raw-json">Raw JSON</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={4} className="gap-4">
              <Card>
                <Text>Version</Text>
                <div className="mt-2">
                  <Title>{currentVersion}</Title>
                  <Badge color="blue" className="mt-1">v{currentVersion}</Badge>
                </div>
              </Card>

              <Card>
                <Text>Prompt Type</Text>
                <div className="mt-2">
                  <Title>{promptData.prompt_info?.prompt_type || "-"}</Title>
                </div>
              </Card>

              <Card>
                <Text>Created By</Text>
                <div className="mt-2">
                  <Title className="text-sm">{promptData.created_by || "-"}</Title>
                </div>
              </Card>

              <Card>
                <Text>Created At</Text>
                <div className="mt-2">
                  <Title className="text-sm">{formatDate(promptData.created_at)}</Title>
                  <Text className="text-xs">Updated: {formatDate(promptData.updated_at)}</Text>
                </div>
              </Card>
            </Grid>

            {/* Version History Table */}
            <Card className="mt-6">
              <Title className="mb-3">Version History — {selectedEnv}</Title>
              {loadingVersions ? (
                <Text>Loading versions...</Text>
              ) : versionHistory.length > 0 ? (
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Version</TableHeaderCell>
                      <TableHeaderCell>Created By</TableHeaderCell>
                      <TableHeaderCell>Date</TableHeaderCell>
                      <TableHeaderCell>Actions</TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {versionHistory.map((v) => {
                      const vNum = v.version || 1;
                      const isSelected = vNum === selectedVersion;
                      const isLatest = vNum === latestVersion;
                      return (
                        <TableRow
                          key={vNum}
                          className={`cursor-pointer hover:bg-blue-50 transition-colors ${
                            isSelected ? "bg-blue-50" : ""
                          }`}
                          onClick={() => handleVersionClick(v)}
                        >
                          <TableCell>
                            <span className={isSelected ? "font-bold" : ""}>
                              v{vNum}
                            </span>
                            {isLatest && (
                              <Badge color="blue" className="ml-2" size="xs">latest</Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <span className="text-sm">{v.created_by || "-"}</span>
                          </TableCell>
                          <TableCell>
                            <span className="text-sm">{formatDate(v.created_at)}</span>
                          </TableCell>
                          <TableCell>
                            <TremorButton
                              icon={PencilIcon}
                              variant="light"
                              size="xs"
                              onClick={(e) => {
                                e.stopPropagation();
                                // Build a response-like object for the editor
                                const editData = {
                                  prompt_spec: {
                                    ...v,
                                    prompt_id: basePromptId,
                                    environment: selectedEnv,
                                  },
                                  raw_prompt_template: isSelected ? promptTemplate : null,
                                };
                                onEdit?.(editData);
                              }}
                            >
                              Edit
                            </TremorButton>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              ) : (
                <Text className="text-gray-400">No versions found in {selectedEnv}</Text>
              )}
            </Card>
          </TabPanel>

          {/* Prompt Template Panel */}
          {promptTemplate && (
            <TabPanel>
              <Card>
                <div className="flex justify-between items-center mb-4">
                  <Title>Prompt Template</Title>
                  <Button
                    type="text"
                    size="small"
                    icon={copiedStates["prompt-content"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                    onClick={() => copyToClipboard(promptTemplate.content, "prompt-content")}
                    className={`transition-all duration-200 ${
                      copiedStates["prompt-content"]
                        ? "text-green-600 bg-green-50 border-green-200"
                        : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                    }`}
                  >
                    {copiedStates["prompt-content"] ? "Copied!" : "Copy Content"}
                  </Button>
                </div>

                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Template ID</Text>
                    <div className="font-mono text-sm bg-gray-50 p-2 rounded">{promptTemplate.litellm_prompt_id}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Content</Text>
                    <div className="mt-2 p-4 bg-gray-50 rounded-md border overflow-auto max-h-96">
                      <pre className="text-sm text-gray-800 whitespace-pre-wrap">{promptTemplate.content}</pre>
                    </div>
                  </div>

                  {promptTemplate.metadata && Object.keys(promptTemplate.metadata).length > 0 && (
                    <div>
                      <Text className="font-medium">Template Metadata</Text>
                      <div className="mt-2 p-3 bg-gray-50 rounded-md border">
                        <pre className="text-xs text-gray-800 whitespace-pre-wrap overflow-auto max-h-64">
                          {JSON.stringify(promptTemplate.metadata, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            </TabPanel>
          )}

          {/* Raw JSON Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Raw API Response</Title>
                <Button
                  type="text"
                  size="small"
                  icon={copiedStates["raw-json"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                  onClick={() => copyToClipboard(JSON.stringify(rawApiResponse, null, 2), "raw-json")}
                  className={`transition-all duration-200 ${
                    copiedStates["raw-json"]
                      ? "text-green-600 bg-green-50 border-green-200"
                      : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  {copiedStates["raw-json"] ? "Copied!" : "Copy JSON"}
                </Button>
              </div>

              <div className="p-4 bg-gray-50 rounded-md border overflow-auto">
                <pre className="text-xs text-gray-800 whitespace-pre-wrap">
                  {JSON.stringify(rawApiResponse, null, 2)}
                </pre>
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Delete Confirmation Modal */}
      <Modal
        title="Delete Prompt"
        open={showDeleteConfirm}
        onOk={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        confirmLoading={isDeleting}
        okText="Delete"
        okButtonProps={{ danger: true }}
      >
        <p>
          Are you sure you want to delete prompt: <strong>{basePromptId}</strong>?
        </p>
        <p>This action cannot be undone.</p>
      </Modal>
    </div>
  );
};

export default PromptInfoView;
