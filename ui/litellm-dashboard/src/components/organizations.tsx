import OrganizationFilters, { FilterState } from "@/app/(dashboard)/organizations/OrganizationFilters";
import { Info as InfoCircleOutlined } from "lucide-react";
import {
  ChevronDown as ChevronDownIcon,
  ChevronRight as ChevronRightIcon,
  RefreshCcw as RefreshIcon,
} from "lucide-react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import React, { useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { formatNumberWithCommas } from "../utils/dataUtils";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import { ModelSelect } from "./ModelSelect/ModelSelect";
import NotificationsManager from "./molecules/notifications_manager";
import { Organization, organizationCreateCall, organizationDeleteCall, organizationListCall } from "./networking";
import OrganizationInfoView from "./organization/organization_view";
import NumericalInput from "./shared/numerical_input";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";

interface OrganizationsTableProps {
  organizations: Organization[];
  userRole: string;
  userModels: string[];
  accessToken: string | null;
  lastRefreshed?: string;
  handleRefreshClick?: () => void;
  currentOrg?: any;
  guardrailsList?: string[];
  setOrganizations: (organizations: Organization[]) => void;
  premiumUser: boolean;
}

interface OrgFormValues {
  organization_alias: string;
  models: string[];
  max_budget: number | null;
  budget_duration: string;
  tpm_limit: number | null;
  rpm_limit: number | null;
  allowed_vector_store_ids: string[];
  allowed_mcp_servers_and_groups: { servers?: string[]; accessGroups?: string[] } | null;
  metadata: string;
}

export const fetchOrganizations = async (
  accessToken: string,
  setOrganizations: (organizations: Organization[]) => void,
  org_id: string | null = null,
  org_alias: string | null = null,
) => {
  const organizations = await organizationListCall(accessToken, org_id, org_alias);
  setOrganizations(organizations);
};

const OrganizationsTable: React.FC<OrganizationsTableProps> = ({
  organizations,
  userRole,
  userModels,
  accessToken,
  lastRefreshed,
  handleRefreshClick,
  currentOrg,
  guardrailsList = [],
  setOrganizations,
  premiumUser,
}) => {
  void userModels;
  void currentOrg;
  void guardrailsList;
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);
  const [editOrg, setEditOrg] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [orgToDelete, setOrgToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isOrgModalVisible, setIsOrgModalVisible] = useState(false);
  const form = useForm<OrgFormValues>({
    defaultValues: {
      organization_alias: "",
      models: [],
      max_budget: null,
      budget_duration: "",
      tpm_limit: null,
      rpm_limit: null,
      allowed_vector_store_ids: [],
      allowed_mcp_servers_and_groups: null,
      metadata: "",
    },
  });
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    org_id: "",
    org_alias: "",
    sort_by: "created_at",
    sort_order: "desc",
  });

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    if (accessToken) {
      organizationListCall(accessToken, newFilters.org_id || null, newFilters.org_alias || null)
        .then((response) => {
          if (response) {
            setOrganizations(response);
          }
        })
        .catch((error) => {
          console.error("Error fetching organizations:", error);
        });
    }
  };

  const handleFilterReset = () => {
    setFilters({
      org_id: "",
      org_alias: "",
      sort_by: "created_at",
      sort_order: "desc",
    });
    if (accessToken) {
      organizationListCall(accessToken, null, null)
        .then((response) => {
          if (response) {
            setOrganizations(response);
          }
        })
        .catch((error) => {
          console.error("Error fetching organizations:", error);
        });
    }
  };

  const handleDelete = (orgId: string | null) => {
    if (!orgId) return;
    setOrgToDelete(orgId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!orgToDelete || !accessToken) return;

    try {
      setIsDeleting(true);
      await organizationDeleteCall(accessToken, orgToDelete);
      NotificationsManager.success("Organization deleted successfully");

      setIsDeleteModalOpen(false);
      setOrgToDelete(null);
      await fetchOrganizations(accessToken, setOrganizations, filters.org_id || null, filters.org_alias || null);
    } catch (error) {
      console.error("Error deleting organization:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setOrgToDelete(null);
  };

  const handleCreate = form.handleSubmit(async (values) => {
    try {
      if (!accessToken) return;

      const payload: Record<string, any> = { ...values };

      if (
        (values.allowed_vector_store_ids && values.allowed_vector_store_ids.length > 0) ||
        (values.allowed_mcp_servers_and_groups &&
          ((values.allowed_mcp_servers_and_groups.servers?.length ?? 0) > 0 ||
            (values.allowed_mcp_servers_and_groups.accessGroups?.length ?? 0) > 0))
      ) {
        payload.object_permission = {};
        if (values.allowed_vector_store_ids && values.allowed_vector_store_ids.length > 0) {
          payload.object_permission.vector_stores = values.allowed_vector_store_ids;
          delete payload.allowed_vector_store_ids;
        }
        if (values.allowed_mcp_servers_and_groups) {
          if ((values.allowed_mcp_servers_and_groups.servers?.length ?? 0) > 0) {
            payload.object_permission.mcp_servers = values.allowed_mcp_servers_and_groups.servers;
          }
          if ((values.allowed_mcp_servers_and_groups.accessGroups?.length ?? 0) > 0) {
            payload.object_permission.mcp_access_groups = values.allowed_mcp_servers_and_groups.accessGroups;
          }
          delete payload.allowed_mcp_servers_and_groups;
        }
      }

      await organizationCreateCall(accessToken, payload);
      NotificationsManager.success("Organization created successfully");
      setIsOrgModalVisible(false);
      form.reset();
      fetchOrganizations(accessToken, setOrganizations, filters.org_id || null, filters.org_alias || null);
    } catch (error) {
      console.error("Error creating organization:", error);
    }
  });

  const handleCancel = () => {
    setIsOrgModalVisible(false);
    form.reset();
  };

  if (!premiumUser) {
    return (
      <div>
        <p className="text-sm">
          This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key{" "}
          <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer">
            here
          </a>
          .
        </p>
      </div>
    );
  }

  return (
    <TooltipProvider>
    <div className="w-full mx-4 h-[75vh]">
      <div className="grid grid-cols-1 gap-2 p-8 w-full mt-2">
        <div className="col-span-1 flex flex-col gap-2">
          {(userRole === "Admin" || userRole === "Org Admin") && (
            <Button className="w-fit" onClick={() => setIsOrgModalVisible(true)}>
              + Create New Organization
            </Button>
          )}
          {selectedOrgId ? (
            <OrganizationInfoView
              organizationId={selectedOrgId}
              onClose={() => {
                setSelectedOrgId(null);
                setEditOrg(false);
              }}
              accessToken={accessToken}
              is_org_admin={true}
              is_proxy_admin={userRole === "Admin"}
              userModels={userModels}
              editOrg={editOrg}
            />
          ) : (
            <Tabs defaultValue="your-organizations" className="gap-2 h-[75vh] w-full">
              <TabsList className="flex justify-between mt-2 w-full items-center">
                <div className="flex">
                  <TabsTrigger value="your-organizations">Your Organizations</TabsTrigger>
                </div>
                <div className="flex items-center space-x-2">
                  {lastRefreshed && (
                    <span className="text-sm text-muted-foreground">
                      Last Refreshed: {lastRefreshed}
                    </span>
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="self-center"
                    onClick={handleRefreshClick}
                    aria-label="Refresh"
                  >
                    <RefreshIcon className="h-4 w-4" />
                  </Button>
                </div>
              </TabsList>
              <TabsContent value="your-organizations">
                <p className="text-sm">
                  Click on &ldquo;Organization ID&rdquo; to view organization details.
                </p>
                <div className="grid grid-cols-1 gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                  <div className="col-span-1">
                    <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                      <div className="border-b border-border px-6 py-4">
                        <div className="flex flex-col space-y-4">
                          <OrganizationFilters
                            filters={filters}
                            showFilters={showFilters}
                            onToggleFilters={setShowFilters}
                            onChange={handleFilterChange}
                            onReset={handleFilterReset}
                          />
                        </div>
                      </div>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableHeaderCell>Organization ID</TableHeaderCell>
                            <TableHeaderCell>Organization Name</TableHeaderCell>
                            <TableHeaderCell>Created</TableHeaderCell>
                            <TableHeaderCell>Spend (USD)</TableHeaderCell>
                            <TableHeaderCell>Budget (USD)</TableHeaderCell>
                            <TableHeaderCell>Models</TableHeaderCell>
                            <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                            <TableHeaderCell>Info</TableHeaderCell>
                            <TableHeaderCell>Actions</TableHeaderCell>
                          </TableRow>
                        </TableHead>

                        <TableBody>
                          {organizations && organizations.length > 0
                            ? organizations
                                .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                                .map((org: Organization) => (
                                  <TableRow key={org.organization_id}>
                                    <TableCell>
                                      <div className="overflow-hidden">
                                        <Tooltip>
                                          <TooltipTrigger asChild>
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              /* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */
                                              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px] h-auto"
                                              onClick={() => setSelectedOrgId(org.organization_id)}
                                            >
                                              {org.organization_id?.slice(0, 7)}
                                              ...
                                            </Button>
                                          </TooltipTrigger>
                                          <TooltipContent>{org.organization_id}</TooltipContent>
                                        </Tooltip>
                                      </div>
                                    </TableCell>
                                    <TableCell>{org.organization_alias}</TableCell>
                                    <TableCell>
                                      {org.created_at ? new Date(org.created_at).toLocaleDateString() : "N/A"}
                                    </TableCell>
                                    <TableCell>{formatNumberWithCommas(org.spend, 4)}</TableCell>
                                    <TableCell>
                                      {org.litellm_budget_table?.max_budget !== null &&
                                      org.litellm_budget_table?.max_budget !== undefined
                                        ? org.litellm_budget_table?.max_budget
                                        : "No limit"}
                                    </TableCell>
                                    <TableCell
                                      style={{
                                        whiteSpace: "pre-wrap",
                                        overflow: "hidden",
                                      }}
                                      className={org.models.length > 3 ? "px-0" : ""}
                                    >
                                      <div className="flex flex-col">
                                        {Array.isArray(org.models) ? (
                                          <div className="flex flex-col">
                                            {org.models.length === 0 ? (
                                              <Badge variant="destructive" className="mb-1 w-fit">
                                                All Proxy Models
                                              </Badge>
                                            ) : (
                                              <div className="flex items-start">
                                                {org.models.length > 3 && (
                                                  <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-5 w-5"
                                                    onClick={() => {
                                                      setExpandedAccordions((prev) => ({
                                                        ...prev,
                                                        [org.organization_id || ""]:
                                                          !prev[org.organization_id || ""],
                                                      }));
                                                    }}
                                                    aria-label="Toggle models"
                                                  >
                                                    {expandedAccordions[org.organization_id || ""] ? (
                                                      <ChevronDownIcon className="h-3 w-3" />
                                                    ) : (
                                                      <ChevronRightIcon className="h-3 w-3" />
                                                    )}
                                                  </Button>
                                                )}
                                                <div className="flex flex-wrap gap-1">
                                                  {org.models.slice(0, 3).map((model, index) =>
                                                    model === "all-proxy-models" ? (
                                                      <Badge key={index} variant="destructive">
                                                        All Proxy Models
                                                      </Badge>
                                                    ) : (
                                                      <Badge key={index} variant="secondary">
                                                        {model.length > 30
                                                          ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                          : getModelDisplayName(model)}
                                                      </Badge>
                                                    ),
                                                  )}
                                                  {org.models.length > 3 &&
                                                    !expandedAccordions[org.organization_id || ""] && (
                                                      <Badge
                                                        variant="outline"
                                                        className="cursor-pointer"
                                                      >
                                                        +{org.models.length - 3}{" "}
                                                        {org.models.length - 3 === 1 ? "more model" : "more models"}
                                                      </Badge>
                                                    )}
                                                  {expandedAccordions[org.organization_id || ""] && (
                                                    <div className="flex flex-wrap gap-1">
                                                      {org.models.slice(3).map((model, index) =>
                                                        model === "all-proxy-models" ? (
                                                          <Badge key={index + 3} variant="destructive">
                                                            All Proxy Models
                                                          </Badge>
                                                        ) : (
                                                          <Badge key={index + 3} variant="secondary">
                                                            {model.length > 30
                                                              ? `${getModelDisplayName(model).slice(0, 30)}...`
                                                              : getModelDisplayName(model)}
                                                          </Badge>
                                                        ),
                                                      )}
                                                    </div>
                                                  )}
                                                </div>
                                              </div>
                                            )}
                                          </div>
                                        ) : null}
                                      </div>
                                    </TableCell>
                                    <TableCell>
                                      <span className="text-sm">
                                        TPM:{" "}
                                        {org.litellm_budget_table?.tpm_limit
                                          ? org.litellm_budget_table?.tpm_limit
                                          : "Unlimited"}
                                        <br />
                                        RPM:{" "}
                                        {org.litellm_budget_table?.rpm_limit
                                          ? org.litellm_budget_table?.rpm_limit
                                          : "Unlimited"}
                                      </span>
                                    </TableCell>
                                    <TableCell>
                                      <span className="text-sm">{org.members?.length || 0} Members</span>
                                    </TableCell>
                                    <TableCell>
                                      {userRole === "Admin" && (
                                        <>
                                          <TableIconActionButton
                                            variant="Edit"
                                            tooltipText="Edit organization"
                                            onClick={() => {
                                              setSelectedOrgId(org.organization_id);
                                              setEditOrg(true);
                                            }}
                                          />
                                          <TableIconActionButton
                                            variant="Delete"
                                            tooltipText="Delete organization"
                                            onClick={() => handleDelete(org.organization_id)}
                                          />
                                        </>
                                      )}
                                    </TableCell>
                                  </TableRow>
                                ))
                            : null}
                        </TableBody>
                      </Table>
                    </Card>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>

      <Dialog
        open={isOrgModalVisible}
        onOpenChange={(o) => {
          if (!o) handleCancel();
        }}
      >
        <DialogContent className="max-w-[800px]">
          <DialogHeader>
            <DialogTitle>Create Organization</DialogTitle>
          </DialogHeader>
          <FormProvider {...form}>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                <Label htmlFor="org-alias" className="mt-2 text-left">
                  Organization Name <span className="text-destructive">*</span>
                </Label>
                <div className="space-y-1">
                  <Input
                    id="org-alias"
                    {...form.register("organization_alias", {
                      required: "Please input an organization name",
                    })}
                  />
                  {form.formState.errors.organization_alias && (
                    <p className="text-sm text-destructive">
                      {form.formState.errors.organization_alias.message as string}
                    </p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                <Label className="mt-2 text-left">Models</Label>
                <Controller
                  control={form.control}
                  name="models"
                  render={({ field }) => (
                    <ModelSelect
                      options={{ showAllProxyModelsOverride: true, includeSpecialOptions: true }}
                      value={field.value}
                      onChange={(values) => field.onChange(values)}
                      context="organization"
                    />
                  )}
                />
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
                <Label className="text-left">Max Budget (USD)</Label>
                <Controller
                  control={form.control}
                  name="max_budget"
                  render={({ field }) => (
                    <NumericalInput
                      step={0.01}
                      width={200}
                      value={field.value as any}
                      onChange={field.onChange}
                    />
                  )}
                />
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
                <Label className="text-left">Reset Budget</Label>
                <Controller
                  control={form.control}
                  name="budget_duration"
                  render={({ field }) => (
                    <Select value={field.value || ""} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue placeholder="n/a" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="24h">daily</SelectItem>
                        <SelectItem value="7d">weekly</SelectItem>
                        <SelectItem value="30d">monthly</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
                <Label className="text-left">Tokens per minute Limit (TPM)</Label>
                <Controller
                  control={form.control}
                  name="tpm_limit"
                  render={({ field }) => (
                    <NumericalInput
                      step={1}
                      width={400}
                      value={field.value as any}
                      onChange={field.onChange}
                    />
                  )}
                />
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-center">
                <Label className="text-left">Requests per minute Limit (RPM)</Label>
                <Controller
                  control={form.control}
                  name="rpm_limit"
                  render={({ field }) => (
                    <NumericalInput
                      step={1}
                      width={400}
                      value={field.value as any}
                      onChange={field.onChange}
                    />
                  )}
                />
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start mt-4">
                <Label className="mt-2 text-left inline-flex items-center gap-1">
                  Allowed Vector Stores
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <InfoCircleOutlined className="h-3.5 w-3.5 ml-1" />
                    </TooltipTrigger>
                    <TooltipContent>
                      Select which vector stores this organization can access by default. Leave
                      empty for access to all vector stores
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <div className="space-y-1">
                  <Controller
                    control={form.control}
                    name="allowed_vector_store_ids"
                    render={({ field }) => (
                      <VectorStoreSelector
                        onChange={(values) => field.onChange(values)}
                        value={field.value}
                        accessToken={accessToken || ""}
                        placeholder="Select vector stores (optional)"
                      />
                    )}
                  />
                  <p className="text-xs text-muted-foreground">
                    Select vector stores this organization can access. Leave empty for access to all
                    vector stores
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start mt-4">
                <Label className="mt-2 text-left inline-flex items-center gap-1">
                  Allowed MCP Servers
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <InfoCircleOutlined className="h-3.5 w-3.5 ml-1" />
                    </TooltipTrigger>
                    <TooltipContent>
                      Select which MCP servers and access groups this organization can access by
                      default.
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <div className="space-y-1">
                  <Controller
                    control={form.control}
                    name="allowed_mcp_servers_and_groups"
                    render={({ field }) => (
                      <MCPServerSelector
                        onChange={(values) => field.onChange(values)}
                        value={field.value as any}
                        accessToken={accessToken || ""}
                        placeholder="Select MCP servers and access groups (optional)"
                      />
                    )}
                  />
                  <p className="text-xs text-muted-foreground">
                    Select MCP servers and access groups this organization can access.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-[1fr_2fr] gap-4 items-start">
                <Label className="mt-2 text-left">Metadata</Label>
                <Textarea rows={4} {...form.register("metadata")} />
              </div>

              <div className="text-right mt-3">
                <Button type="submit">Create Organization</Button>
              </div>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Organization?"
        message="Are you sure you want to delete this organization? This action cannot be undone."
        resourceInformationTitle="Organization Information"
        resourceInformation={[{ label: "Organization ID", value: orgToDelete, code: true }]}
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeleting}
      />
    </div>
    </TooltipProvider>
  );
};

export default OrganizationsTable;
