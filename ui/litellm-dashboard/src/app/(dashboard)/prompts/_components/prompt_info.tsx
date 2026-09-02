import React, { useState, useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  getPromptInfo,
  getPromptVersions,
  PromptSpec,
  PromptTemplateBase,
  deletePromptCall,
} from "@/components/networking";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { ArrowLeft, CheckIcon, CopyIcon, Pencil, Trash2 } from "lucide-react";
import { toast } from "@/lib/toast";
import PromptCodeSnippets from "./prompt_editor_view/PromptCodeSnippets";
import { extractModel, extractTemplateVariables, getBasePromptId, getCurrentVersion } from "./prompt_utils";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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
      toast.fromError("Failed to load prompt information");
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
      toast.success(`Prompt "${basePromptId}" deleted successfully`);
      onDelete?.();
      onClose();
    } catch (error) {
      console.error("Error deleting prompt:", error);
      toast.fromError("Failed to delete prompt");
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
      toast.fromError(`Failed to load version v${versionNum}`);
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
          <ArrowLeft className="size-4" />
          Back to Prompts
        </Button>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h1 className="text-2xl font-semibold">Prompt Details</h1>
            <div className="flex items-center cursor-pointer">
              <p className="text-sm text-muted-foreground font-mono">{basePromptId}</p>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={() => copyToClipboard(basePromptId, "prompt-id")}
                className={`left-2 z-raised transition-all duration-200 ${
                  copiedStates["prompt-id"]
                    ? "text-success bg-success/10 border-success/20"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
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
              <Pencil />
              Prompt Studio
            </Button>
            {isAdmin && (
              <Button variant="secondary" onClick={handleDeleteClick} className="flex items-center">
                <Trash2 />
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
                      ? "bg-destructive/15 text-destructive border-2 border-destructive/30"
                      : env === "staging"
                        ? "bg-warning/15 text-warning border-2 border-warning/30"
                        : "bg-success/15 text-success border-2 border-success/30"
                    : "bg-muted text-muted-foreground border-2 border-transparent hover:bg-accent"
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
        <div className="mb-4 p-3 bg-warning/10 border border-warning/20 rounded-lg flex items-center justify-between">
          <p className="text-sm text-warning">
            Viewing v{selectedVersion} — not the latest version (v{latestVersion})
          </p>
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

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          {promptTemplate && (
            <TabsTrigger value="prompt-template" className="flex-none rounded-none px-4 py-2">
              Prompt Template
            </TabsTrigger>
          )}
          <TabsTrigger value="raw-json" className="flex-none rounded-none px-4 py-2">
            Raw JSON
          </TabsTrigger>
        </TabsList>

        <div>
          {/* Overview Panel */}
          <TabsContent value="overview" keepMounted>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="block p-6">
                <p>Version</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">{currentVersion}</h3>
                  <Badge variant="secondary" className="mt-1">
                    v{currentVersion}
                  </Badge>
                </div>
              </Card>

              <Card className="block p-6">
                <p>Prompt Type</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">{promptData.prompt_info?.prompt_type || "-"}</h3>
                </div>
              </Card>

              <Card className="block p-6">
                <p>Created By</p>
                <div className="mt-2">
                  <h3 className="text-sm font-medium">{promptData.created_by || "-"}</h3>
                </div>
              </Card>

              <Card className="block p-6">
                <p>Created At</p>
                <div className="mt-2">
                  <h3 className="text-sm font-medium">{formatDate(promptData.created_at)}</h3>
                  <p className="text-xs">Updated: {formatDate(promptData.updated_at)}</p>
                </div>
              </Card>
            </div>

            {/* Version History Table */}
            <Card className="block mt-6 p-6">
              <h3 className="text-lg font-medium mb-3">Version History — {selectedEnv}</h3>
              {loadingVersions ? (
                <p>Loading versions...</p>
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
                          className={`cursor-pointer hover:bg-info/10 transition-colors ${
                            isSelected ? "bg-info/10" : ""
                          }`}
                          onClick={() => handleVersionClick(v)}
                        >
                          <TableCell>
                            <span className={isSelected ? "font-bold" : ""}>v{vNum}</span>
                            {isLatest && (
                              <Badge variant="secondary" className="ml-2">
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
                              <Pencil />
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
            <TabsContent value="prompt-template" keepMounted>
              <Card className="block p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Prompt Template</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(promptTemplate.content, "prompt-content")}
                    className={`transition-all duration-200 ${
                      copiedStates["prompt-content"]
                        ? "text-success bg-success/10 border-success/20"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent"
                    }`}
                  >
                    {copiedStates["prompt-content"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                    {copiedStates["prompt-content"] ? "Copied!" : "Copy Content"}
                  </Button>
                </div>

                <div className="space-y-4">
                  <div>
                    <p className="font-medium">Template ID</p>
                    <div className="font-mono text-sm bg-muted p-2 rounded-sm">{promptTemplate.litellm_prompt_id}</div>
                  </div>

                  <div>
                    <p className="font-medium">Content</p>
                    <div className="mt-2 p-4 bg-muted rounded-md border overflow-auto max-h-96">
                      <pre className="text-sm text-foreground whitespace-pre-wrap">{promptTemplate.content}</pre>
                    </div>
                  </div>

                  {promptTemplate.metadata && Object.keys(promptTemplate.metadata).length > 0 && (
                    <div>
                      <p className="font-medium">Template Metadata</p>
                      <div className="mt-2 p-3 bg-muted rounded-md border">
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
          <TabsContent value="raw-json" keepMounted>
            <Card className="block p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Raw API Response</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(JSON.stringify(rawApiResponse, null, 2), "raw-json")}
                  className={`transition-all duration-200 ${
                    copiedStates["raw-json"]
                      ? "text-success bg-success/10 border-success/20"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent"
                  }`}
                >
                  {copiedStates["raw-json"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                  {copiedStates["raw-json"] ? "Copied!" : "Copy JSON"}
                </Button>
              </div>

              <div className="p-4 bg-muted rounded-md border overflow-auto">
                <pre className="text-xs text-foreground whitespace-pre-wrap">
                  {JSON.stringify(rawApiResponse, null, 2)}
                </pre>
              </div>
            </Card>
          </TabsContent>
        </div>
      </Tabs>

      {/* Delete Confirmation Modal */}
      <Dialog open={showDeleteConfirm} onOpenChange={(open) => !open && handleDeleteCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Delete Prompt</DialogTitle>
          </DialogHeader>
          <p>
            Are you sure you want to delete prompt: <strong>{basePromptId}</strong>?
          </p>
          <p>This action cannot be undone.</p>
          <DialogFooter>
            <Button variant="outline" onClick={handleDeleteCancel}>
              Cancel
            </Button>
            <Button onClick={handleDeleteConfirm} variant="destructive" disabled={isDeleting} aria-busy={isDeleting}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PromptInfoView;
