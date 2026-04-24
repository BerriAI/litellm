import React, { useState, useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ArrowLeft as ArrowLeftIcon,
  Trash2 as TrashIcon,
  Pencil as PencilIcon,
  Check as CheckIcon,
  Copy as CopyIcon,
} from "lucide-react";
import {
  getPromptInfo,
  getPromptVersions,
  PromptSpec,
  PromptTemplateBase,
  deletePromptCall,
} from "@/components/networking";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import NotificationsManager from "../molecules/notifications_manager";
import PromptCodeSnippets from "./prompt_editor_view/PromptCodeSnippets";
import {
  extractModel,
  extractTemplateVariables,
  getBasePromptId,
  getCurrentVersion,
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

  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
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
  const latestVersion = versionHistory.length > 0 ? Math.max(...versionHistory.map((v) => v.version || 1)) : null;
  const isViewingOldVersion = latestVersion !== null && selectedVersion !== null && selectedVersion < latestVersion;

  return (
    <div className="p-4">
      <div>
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeftIcon className="h-4 w-4" />
          Back to Prompts
        </Button>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-semibold m-0">Prompt Details</h2>
            <div className="flex items-center cursor-pointer">
              <span className="text-muted-foreground font-mono">{basePromptId}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(basePromptId, "prompt-id")}
                className={`left-2 z-10 transition-all duration-200 ${
                  copiedStates["prompt-id"]
                    ? "text-green-600 bg-green-50 border-green-200"
                    : "text-muted-foreground hover:text-foreground"
                }`}
                aria-label="Copy prompt ID"
              >
                {copiedStates["prompt-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              </Button>
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
            <Button onClick={() => onEdit?.(rawApiResponse)} className="flex items-center">
              <PencilIcon className="h-4 w-4" />
              Prompt Studio
            </Button>
            {isAdmin && (
              <Button variant="secondary" onClick={handleDeleteClick} className="flex items-center">
                <TrashIcon className="h-4 w-4" />
                Delete Prompt
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Environment Tabs */}
      {environments.length > 0 && (
        <div className="flex gap-2 mb-4">
          {[...environments]
            .sort((a, b) => {
              const order: Record<string, number> = { development: 0, staging: 1, production: 2 };
              return (order[a] ?? 99) - (order[b] ?? 99);
            })
            .map((env) => (
              <button
                key={env}
                onClick={() => {
                  setSelectedEnv(env);
                  setSelectedVersion(null);
                }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedEnv === env
                    ? env === "production"
                      ? "bg-red-100 text-red-800 border-2 border-red-300 dark:bg-red-950 dark:text-red-300"
                      : env === "staging"
                        ? "bg-yellow-100 text-yellow-800 border-2 border-yellow-300 dark:bg-yellow-950 dark:text-yellow-300"
                        : "bg-green-100 text-green-800 border-2 border-green-300 dark:bg-green-950 dark:text-green-300"
                    : "bg-muted text-muted-foreground border-2 border-transparent hover:bg-muted/80"
                }`}
              >
                {env}
                {versionHistory.length > 0 && selectedEnv === env && (
                  <span className="ml-1 text-xs opacity-75">(v{latestVersion})</span>
                )}
              </button>
            ))}
        </div>
      )}

      {/* Old version banner */}
      {isViewingOldVersion && (
        <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center justify-between dark:bg-amber-950 dark:border-amber-900">
          <span className="text-amber-800 dark:text-amber-200">
            Viewing v{selectedVersion} — not the latest version (v{latestVersion})
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const latest = versionHistory.find((v) => v.version === latestVersion);
              if (latest) handleVersionClick(latest);
            }}
          >
            Go to latest
          </Button>
        </div>
      )}

      <Tabs defaultValue="overview" className="mt-1">
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {promptTemplate && <TabsTrigger value="prompt-template">Prompt Template</TabsTrigger>}
          <TabsTrigger value="raw-json">Raw JSON</TabsTrigger>
        </TabsList>

        {/* Overview Panel */}
        <TabsContent value="overview">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-6">
              <p className="text-sm text-muted-foreground">Version</p>
              <div className="mt-2">
                <h3 className="text-lg font-semibold">{currentVersion}</h3>
                <Badge variant="secondary" className="mt-1">
                  v{currentVersion}
                </Badge>
              </div>
            </Card>

            <Card className="p-6">
              <p className="text-sm text-muted-foreground">Prompt Type</p>
              <div className="mt-2">
                <h3 className="text-lg font-semibold">{promptData.prompt_info?.prompt_type || "-"}</h3>
              </div>
            </Card>

            <Card className="p-6">
              <p className="text-sm text-muted-foreground">Created By</p>
              <div className="mt-2">
                <h3 className="text-sm font-semibold">{promptData.created_by || "-"}</h3>
              </div>
            </Card>

            <Card className="p-6">
              <p className="text-sm text-muted-foreground">Created At</p>
              <div className="mt-2">
                <h3 className="text-sm font-semibold">{formatDate(promptData.created_at)}</h3>
                <p className="text-xs text-muted-foreground">Updated: {formatDate(promptData.updated_at)}</p>
              </div>
            </Card>
          </div>

          {/* Version History Table */}
          <Card className="mt-6 p-6">
            <h3 className="text-lg font-semibold mb-3">Version History — {selectedEnv}</h3>
            {loadingVersions ? (
              <p className="text-sm text-muted-foreground">Loading versions...</p>
            ) : versionHistory.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Version</TableHead>
                    <TableHead>Created By</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {versionHistory.map((v) => {
                    const vNum = v.version || 1;
                    const isSelected = vNum === selectedVersion;
                    const isLatest = vNum === latestVersion;
                    return (
                      <TableRow
                        key={vNum}
                        className={`cursor-pointer hover:bg-muted transition-colors ${
                          isSelected ? "bg-muted" : ""
                        }`}
                        onClick={() => handleVersionClick(v)}
                      >
                        <TableCell>
                          <span className={isSelected ? "font-bold" : ""}>v{vNum}</span>
                          {isLatest && (
                            <Badge variant="secondary" className="ml-2 text-xs">
                              latest
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <span className="text-sm">{v.created_by || "-"}</span>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm">{formatDate(v.created_at)}</span>
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
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
                            <PencilIcon className="h-4 w-4" />
                            Edit
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            ) : (
              <p className="text-muted-foreground">No versions found in {selectedEnv}</p>
            )}
          </Card>
        </TabsContent>

        {/* Prompt Template Panel */}
        {promptTemplate && (
          <TabsContent value="prompt-template">
            <Card className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Prompt Template</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(promptTemplate.content, "prompt-content")}
                  className={`transition-all duration-200 ${
                    copiedStates["prompt-content"]
                      ? "text-green-600 bg-green-50 border-green-200"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {copiedStates["prompt-content"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                  {copiedStates["prompt-content"] ? "Copied!" : "Copy Content"}
                </Button>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="font-medium">Template ID</p>
                  <div className="font-mono text-sm bg-muted p-2 rounded">{promptTemplate.litellm_prompt_id}</div>
                </div>

                <div>
                  <p className="font-medium">Content</p>
                  <div className="mt-2 p-4 bg-muted rounded-md border border-border overflow-auto max-h-96">
                    <pre className="text-sm text-foreground whitespace-pre-wrap">{promptTemplate.content}</pre>
                  </div>
                </div>

                {promptTemplate.metadata && Object.keys(promptTemplate.metadata).length > 0 && (
                  <div>
                    <p className="font-medium">Template Metadata</p>
                    <div className="mt-2 p-3 bg-muted rounded-md border border-border">
                      <pre className="text-xs text-foreground whitespace-pre-wrap overflow-auto max-h-64">
                        {JSON.stringify(promptTemplate.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </TabsContent>
        )}

        {/* Raw JSON Panel */}
        <TabsContent value="raw-json">
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Raw API Response</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(JSON.stringify(rawApiResponse, null, 2), "raw-json")}
                className={`transition-all duration-200 ${
                  copiedStates["raw-json"]
                    ? "text-green-600 bg-green-50 border-green-200"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {copiedStates["raw-json"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                {copiedStates["raw-json"] ? "Copied!" : "Copy JSON"}
              </Button>
            </div>

            <div className="p-4 bg-muted rounded-md border border-border overflow-auto">
              <pre className="text-xs text-foreground whitespace-pre-wrap">
                {JSON.stringify(rawApiResponse, null, 2)}
              </pre>
            </div>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Delete Confirmation Modal */}
      <Dialog
        open={showDeleteConfirm}
        onOpenChange={(o) => (!o ? handleDeleteCancel() : undefined)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Prompt</DialogTitle>
          </DialogHeader>
          <p>
            Are you sure you want to delete prompt: <strong>{basePromptId}</strong>?
          </p>
          <p>This action cannot be undone.</p>
          <DialogFooter>
            <Button variant="outline" onClick={handleDeleteCancel} disabled={isDeleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
              {isDeleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PromptInfoView;
