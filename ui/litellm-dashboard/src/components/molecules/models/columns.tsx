import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Icon } from "@tremor/react";
import { Tooltip, Checkbox, Switch, Modal } from "antd";
import { getProviderLogoAndName } from "../../provider_info_helpers";
import { ModelData } from "../../model_dashboard/types";
import { TrashIcon, PencilIcon, PencilAltIcon, KeyIcon, ExclamationIcon } from "@heroicons/react/outline";
import DeleteModelButton from "../../delete_model_button";
import { useState } from "react";
import { modelDeleteCall } from "../../networking";
import NotificationsManager from "../notifications_manager";

export const columns = (
  userRole: string,
  userID: string,
  premiumUser: boolean,
  setSelectedModelId: (id: string) => void,
  setSelectedTeamId: (id: string) => void,
  getDisplayModelName: (model: any) => string,
  handleEditClick: (model: any) => void,
  handleRefreshClick: () => void,
  setEditModel: (edit: boolean) => void,
  expandedRows: Set<string>,
  setExpandedRows: (expandedRows: Set<string>) => void,
  // New props for bulk operations
  selectedModels: Set<string>,
  setSelectedModels: (models: Set<string>) => void,
  handleBulkDisable: (modelIds: string[], disable: boolean) => void,
  disabledModels: Set<string>,
  // New prop for access token to handle deletion
  accessToken: string,
): ColumnDef<ModelData>[] => [
  // Checkbox column for bulk selection
  {
    id: "select",
    header: ({ table }) => {
      const allVisibleRows = table.getRowModel().rows;
      const allVisibleModelIds = allVisibleRows.map((row) => row.original.model_info.id);
      const visibleSelectedCount = allVisibleModelIds.filter(id => selectedModels.has(id)).length;
      const isAllSelected = allVisibleModelIds.length > 0 && visibleSelectedCount === allVisibleModelIds.length;
      const isIndeterminate = visibleSelectedCount > 0 && visibleSelectedCount < allVisibleModelIds.length;

      return (
        <Tooltip title={isAllSelected ? "Deselect all" : "Select all"}>
          <Checkbox
            checked={isAllSelected}
            indeterminate={isIndeterminate}
            onChange={(e) => {
              if (e.target.checked) {
                // Select all visible rows
                const newSelected = new Set([...selectedModels, ...allVisibleModelIds]);
                setSelectedModels(newSelected);
              } else {
                // Deselect all visible rows
                const newSelected = new Set([...selectedModels].filter(id => !allVisibleModelIds.includes(id)));
                setSelectedModels(newSelected);
              }
            }}
          />
        </Tooltip>
      );
    },
    cell: ({ row }) => {
      const modelId = row.original.model_info.id;
      return (
        <Checkbox
          checked={selectedModels.has(modelId)}
          onChange={(e) => {
            const newSelected = new Set(selectedModels);
            if (e.target.checked) {
              newSelected.add(modelId);
            } else {
              newSelected.delete(modelId);
            }
            setSelectedModels(newSelected);
          }}
        />
      );
    },
    size: 24,
    enableSorting: false,
    enableResizing: false,
  },
  {
    header: () => <span className="text-sm font-semibold">Model ID</span>,
    accessorKey: "model_info.id",
    cell: ({ row }) => {
      const model = row.original;
      const isDisabled = disabledModels.has(model.model_info.id);
      return (
        <Tooltip title={model.model_info.id}>
          <div 
            className={`font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch] ${
              isDisabled ? 'opacity-50' : ''
            }`}
            onClick={() => setSelectedModelId(model.model_info.id)}
          >
            {model.model_info.id}
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Model Information</span>,
    accessorKey: "model_name",
    size: 250, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const displayName = getDisplayModelName(row.original) || "-";
      const isDisabled = disabledModels.has(model.model_info.id);
      const tooltipContent = (
        <div>
          <div><strong>Provider:</strong> {model.provider || "-"}</div>
          <div><strong>Public Model Name:</strong> {displayName}</div>
          <div><strong>LiteLLM Model Name:</strong> {model.litellm_model_name || "-"}</div>
          {isDisabled && <div><strong>Status:</strong> <span className="text-red-600">Disabled</span></div>}
        </div>
      );
      
      return (
        <Tooltip title={tooltipContent}>
          <div className={`flex items-start space-x-2 min-w-0 w-full max-w-[250px] ${isDisabled ? 'opacity-50' : ''}`}>
            {/* Provider Icon */}
            <div className="flex-shrink-0 mt-0.5">
              {model.provider ? (
                <img
                  src={getProviderLogoAndName(model.provider).logo}
                  alt={`${model.provider} logo`}
                  className="w-4 h-4"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    const parent = target.parentElement;
                    if (parent) {
                      const fallbackDiv = document.createElement('div');
                      fallbackDiv.className = 'w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                      fallbackDiv.textContent = model.provider?.charAt(0) || '-';
                      parent.replaceChild(fallbackDiv, target);
                    }
                  }}
                />
              ) : (
                <div className="w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs">
                  -
                </div>
              )}
            </div>
            
            {/* Model Names Container */}
            <div className="flex flex-col min-w-0 flex-1">
              {/* Public Model Name */}
              <div className="text-xs font-medium text-gray-900 truncate max-w-[210px]">
                {displayName}
              </div>
              {/* LiteLLM Model Name */}
              <div className="text-xs text-gray-500 truncate mt-0.5 max-w-[210px]">
                {model.litellm_model_name || "-"}
              </div>
            </div>
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Credentials</span>,
    accessorKey: "litellm_credential_name",
    size: 180, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const credentialName = model.litellm_params?.litellm_credential_name;
      const isDisabled = disabledModels.has(model.model_info.id);
      
      return credentialName ? (
        <Tooltip title={`Credential: ${credentialName}`}>
          <div className={`flex items-center space-x-2 max-w-[180px] ${isDisabled ? 'opacity-50' : ''}`}>
            <KeyIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
            <span className="text-xs truncate" title={credentialName}>
              {credentialName}
            </span>
          </div>
        </Tooltip>
      ) : (
        <div className={`flex items-center space-x-2 max-w-[180px] ${isDisabled ? 'opacity-50' : ''}`}>
          <KeyIcon className="w-4 h-4 text-gray-300 flex-shrink-0" />
          <span className="text-xs text-gray-400">No credentials</span>
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Created By</span>,
    accessorKey: "model_info.created_by",
    sortingFn: "datetime",
    size: 160, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const createdBy = model.model_info.created_by;
      const createdAt = model.model_info.created_at 
        ? new Date(model.model_info.created_at).toLocaleDateString() 
        : null;
      const isDisabled = disabledModels.has(model.model_info.id);
      
      return (
        <div className={`flex flex-col min-w-0 max-w-[160px] ${isDisabled ? 'opacity-50' : ''}`}>
          {/* Created By - Primary */}
          <div className="text-xs font-medium text-gray-900 truncate" title={createdBy || "Unknown"}>
            {createdBy || "Unknown"}
          </div>
          {/* Created At - Secondary */}
          <div className="text-xs text-gray-500 truncate mt-0.5" title={createdAt || "Unknown date"}>
            {createdAt || "Unknown date"}
          </div>
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Updated At</span>,
    accessorKey: "model_info.updated_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const model = row.original;
      const isDisabled = disabledModels.has(model.model_info.id);
      return (
        <span className={`text-xs ${isDisabled ? 'opacity-50' : ''}`}>
          {model.model_info.updated_at ? new Date(model.model_info.updated_at).toLocaleDateString() : "-"}
        </span>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Costs</span>,
    accessorKey: "input_cost",
    size: 120, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const inputCost = model.input_cost;
      const outputCost = model.output_cost;
      const isDisabled = disabledModels.has(model.model_info.id);
      
      // If both costs are missing or undefined, show "-"
      if (!inputCost && !outputCost) {
        return (
          <div className={`max-w-[120px] ${isDisabled ? 'opacity-50' : ''}`}>
            <span className="text-xs text-gray-400">-</span>
          </div>
        );
      }
      
      return (
        <Tooltip title="Cost per 1M tokens">
          <div className={`flex flex-col min-w-0 max-w-[120px] ${isDisabled ? 'opacity-50' : ''}`}>
            {/* Input Cost - Primary */}
            {inputCost && (
              <div className="text-xs font-medium text-gray-900 truncate">
                In: ${inputCost}
              </div>
            )}
            {/* Output Cost - Secondary */}
            {outputCost && (
              <div className="text-xs text-gray-500 truncate mt-0.5">
                Out: ${outputCost}
              </div>
            )}
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Team ID</span>,
    accessorKey: "model_info.team_id",
    cell: ({ row }) => {
      const model = row.original;
      const isDisabled = disabledModels.has(model.model_info.id);
      return model.model_info.team_id ? (
        <div className={`overflow-hidden ${isDisabled ? 'opacity-50' : ''}`}>
          <Tooltip title={model.model_info.team_id}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => setSelectedTeamId(model.model_info.team_id)}
            >
              {model.model_info.team_id.slice(0, 7)}...
            </Button>
          </Tooltip>
        </div>
      ) : (
        <span className={isDisabled ? 'opacity-50' : ''}>-</span>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Model Access Group</span>,
    accessorKey: "model_info.model_access_group",
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original;
      const accessGroups = model.model_info.access_groups;
      const isDisabled = disabledModels.has(model.model_info.id);
      
      if (!accessGroups || accessGroups.length === 0) {
        return <span className={isDisabled ? 'opacity-50' : ''}>-</span>;
      }
      
      const modelId = model.model_info.id;
      const isExpanded = expandedRows.has(modelId);
      const shouldShowExpandButton = accessGroups.length > 1;
      
      const toggleExpanded = () => {
        const newExpanded = new Set(expandedRows);
        if (isExpanded) {
          newExpanded.delete(modelId);
        } else {
          newExpanded.add(modelId);
        }
        setExpandedRows(newExpanded);
      };
      
      return (
        <div className={`flex items-center gap-1 overflow-hidden ${isDisabled ? 'opacity-50' : ''}`}>
          <Badge
            size="xs"
            color="blue"
            className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
          >
            {accessGroups[0]}
          </Badge>
          
          {(isExpanded || (!shouldShowExpandButton && accessGroups.length === 2)) && 
            accessGroups.slice(1).map((group: string, index: number) => (
              <Badge
                key={index + 1}
                size="xs"
                color="blue"
                className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
              >
                {group}
              </Badge>
            ))
          }
          
          {shouldShowExpandButton && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded();
              }}
              className="text-xs text-blue-600 hover:text-blue-800 px-1 py-0.5 rounded hover:bg-blue-50 h-5 leading-tight flex-shrink-0 whitespace-nowrap"
            >
              {isExpanded ? '−' : `+${accessGroups.length - 1}`}
            </button>
          )}
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Status</span>,
    accessorKey: "model_info.db_model",
    cell: ({ row }) => {
      const model = row.original;
      const isDisabled = disabledModels.has(model.model_info.id);
      return (
        <div className="flex flex-col space-y-1">
          <div className={`
            inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
            ${model.model_info.db_model 
              ? 'bg-blue-50 text-blue-600' 
              : 'bg-gray-100 text-gray-600'}
            ${isDisabled ? 'opacity-50' : ''}
          `}>
            {model.model_info.db_model ? "DB Model" : "Config Model"}
          </div>
          {isDisabled && (
            <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-600">
              Disabled
            </div>
          )}
        </div>
      );
    },
  },
  {
    id: "actions",
    header: "",
    cell: ({ row }) => {
      const model = row.original;
      const canEditModel = userRole === "Admin" || model.model_info?.created_by === userID;
      const isDisabled = disabledModels.has(model.model_info.id);
      const isSelected = selectedModels.has(model.model_info.id);
      const isInBulkMode = selectedModels.size > 0; // Check if any models are selected (bulk mode)
      
      // State for delete confirmation modal
      const [showDeleteModal, setShowDeleteModal] = useState(false);
      const [isDeleting, setIsDeleting] = useState(false);
      const [deleteConfirmInput, setDeleteConfirmInput] = useState("");
      
      // State for disable/enable confirmation modal
      const [showDisableModal, setShowDisableModal] = useState(false);
      const [pendingDisableAction, setPendingDisableAction] = useState<{action: 'disable' | 'enable', checked: boolean} | null>(null);
      
      // Get the model name for confirmation
      const modelNameForConfirmation = getDisplayModelName(model) || model.model_name || model.model_info.id;
      const isDeleteValid = deleteConfirmInput === modelNameForConfirmation;
      
      const handleDelete = async () => {
        if (!canEditModel || !isDeleteValid) return;
        
        setIsDeleting(true);
        try {
          await modelDeleteCall(accessToken, model.model_info.id);
          NotificationsManager.success(`Model ${model.model_info.id} deleted successfully`);
          setShowDeleteModal(false);
          setDeleteConfirmInput("");
          // Remove from selected models if it was selected
          if (selectedModels.has(model.model_info.id)) {
            const newSelected = new Set(selectedModels);
            newSelected.delete(model.model_info.id);
            setSelectedModels(newSelected);
          }
          // Refresh the table
          setTimeout(handleRefreshClick, 1000);
        } catch (error) {
          console.error("Error deleting model:", error);
          NotificationsManager.error("Failed to delete model");
        } finally {
          setIsDeleting(false);
        }
      };

      const handleModalCancel = () => {
        setShowDeleteModal(false);
        setDeleteConfirmInput("");
      };

      const handleDisableToggle = (checked: boolean) => {
        if (!canEditModel || isInBulkMode) return;
        
        const action = checked ? 'enable' : 'disable';
        setPendingDisableAction({ action, checked });
        setShowDisableModal(true);
      };

      const confirmDisableAction = () => {
        if (pendingDisableAction && canEditModel) {
          handleBulkDisable([model.model_info.id], !pendingDisableAction.checked);
        }
        setShowDisableModal(false);
        setPendingDisableAction(null);
      };

      const cancelDisableAction = () => {
        setShowDisableModal(false);
        setPendingDisableAction(null);
      };
      
      return (
        <>
          <div className="flex items-center justify-end gap-2 pr-4">
            {/* Disable/Enable Toggle */}
            <Tooltip title={
              isInBulkMode 
                ? "Individual actions disabled in bulk selection mode" 
                : isDisabled 
                  ? "Enable model" 
                  : "Disable model"
            }>
              <Switch
                size="small"
                checked={!isDisabled}
                onChange={handleDisableToggle}
                disabled={!canEditModel || isInBulkMode}
                className={(!canEditModel || isInBulkMode) ? "opacity-50" : ""}
              />
            </Tooltip>
            
            {/* Delete Button */}
            <Tooltip title={
              isInBulkMode 
                ? "Individual actions disabled in bulk selection mode" 
                : "Delete model"
            }>
              <Icon
                icon={TrashIcon}
                size="sm"
                onClick={() => {
                  if (canEditModel && !isInBulkMode) {
                    setShowDeleteModal(true);
                  }
                }}
                className={(!canEditModel || isInBulkMode) ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:text-red-600"}
              />
            </Tooltip>
          </div>

          {/* Disable/Enable Confirmation Modal */}
          <Modal
            title={
              <div className="flex items-center">
                <ExclamationIcon className="h-5 w-5 text-orange-600 mr-2" />
                {pendingDisableAction?.action === 'disable' ? 'Disable Model' : 'Enable Model'}
              </div>
            }
            open={showDisableModal}
            onCancel={cancelDisableAction}
            footer={null}
            destroyOnClose={true}
            maskClosable={false}
            width={500}
          >
            <div className="mt-4">
              <div className={`border rounded-md p-4 mb-4 ${
                pendingDisableAction?.action === 'disable' 
                  ? 'bg-orange-50 border-orange-200' 
                  : 'bg-green-50 border-green-200'
              }`}>
                <div className="flex">
                  <div className="ml-3">
                    <h3 className={`text-sm font-medium ${
                      pendingDisableAction?.action === 'disable' 
                        ? 'text-orange-800' 
                        : 'text-green-800'
                    }`}>
                      {pendingDisableAction?.action === 'disable' 
                        ? 'This will disable the model' 
                        : 'This will enable the model'}
                    </h3>
                    <div className={`mt-2 text-sm ${
                      pendingDisableAction?.action === 'disable' 
                        ? 'text-orange-700' 
                        : 'text-green-700'
                    }`}>
                      <p>
                        {pendingDisableAction?.action === 'disable' 
                          ? 'The model will no longer be available for API requests.'
                          : 'The model will become available for API requests.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-900 mb-2">Model Details:</p>
                <div className="bg-gray-50 rounded-md p-3 text-sm">
                  <div><strong>Model ID:</strong> {model.model_info.id}</div>
                  <div><strong>Model Name:</strong> {modelNameForConfirmation}</div>
                  <div><strong>Provider:</strong> {model.provider || "-"}</div>
                  <div><strong>Current Status:</strong> 
                    <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-medium ${
                      isDisabled 
                        ? 'bg-red-100 text-red-700' 
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {isDisabled ? 'Disabled' : 'Enabled'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <Button
                  size="xs"
                  variant="secondary"
                  onClick={cancelDisableAction}
                >
                  Cancel
                </Button>
                <Button
                  size="xs"
                  color={pendingDisableAction?.action === 'disable' ? 'orange' : 'green'}
                  onClick={confirmDisableAction}
                >
                  {pendingDisableAction?.action === 'disable' ? 'Disable Model' : 'Enable Model'}
                </Button>
              </div>
            </div>
          </Modal>

          {/* Delete Confirmation Modal */}
          <Modal
            title={
              <div className="flex items-center">
                <ExclamationIcon className="h-6 w-6 text-red-600 mr-2" />
                Delete Model
              </div>
            }
            open={showDeleteModal}
            onCancel={handleModalCancel}
            footer={null}
            destroyOnClose={true}
            maskClosable={false}
            width={600}
          >
            <div className="mt-4">
              <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
                <div className="flex">
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      This action cannot be undone
                    </h3>
                    <div className="mt-2 text-sm text-red-700">
                      <p>
                        This will permanently delete the model and all its configuration.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-900 mb-2">Model Details:</p>
                <div className="bg-gray-50 rounded-md p-3 text-sm">
                  <div><strong>Model ID:</strong> {model.model_info.id}</div>
                  <div><strong>Model Name:</strong> {modelNameForConfirmation}</div>
                  <div><strong>Provider:</strong> {model.provider || "-"}</div>
                </div>
              </div>

              {/* Confirmation Input */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  To confirm deletion, type the model name: <span className="font-mono text-blue-600">{modelNameForConfirmation}</span>
                </label>
                <input
                  type="text"
                  value={deleteConfirmInput}
                  onChange={(e) => setDeleteConfirmInput(e.target.value)}
                  placeholder={`Enter "${modelNameForConfirmation}" to confirm`}
                  className={`w-full px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                    deleteConfirmInput.length > 0 && !isDeleteValid 
                      ? 'border-red-300 bg-red-50' 
                      : 'border-gray-300'
                  }`}
                  disabled={isDeleting}
                  autoFocus
                />
                {deleteConfirmInput.length > 0 && !isDeleteValid && (
                  <p className="mt-1 text-sm text-red-600">
                    Model name doesn't match. Please type exactly: {modelNameForConfirmation}
                  </p>
                )}
              </div>

              <div className="flex justify-end space-x-3">
                <Button
                  size="xs"
                  variant="secondary"
                  onClick={handleModalCancel}
                  disabled={isDeleting}
                >
                  Cancel
                </Button>
                <Button
                  size="xs"
                  color="red"
                  onClick={handleDelete}
                  loading={isDeleting}
                  disabled={isDeleting || !isDeleteValid}
                >
                  {isDeleting ? 'Deleting...' : 'Delete Model'}
                </Button>
              </div>
            </div>
          </Modal>
        </>
      );
    },
  },
];