import React, { useCallback, useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
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
import { ArrowLeft, Info } from "lucide-react";
import {
  credentialListCall,
  CredentialItem,
  vectorStoreInfoCall,
  vectorStoreUpdateCall,
} from "../networking";
import { VectorStore } from "./types";
import {
  Providers,
  providerLogoMap,
  provider_map,
} from "../provider_info_helpers";
import VectorStoreTester from "./VectorStoreTester";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreInfoViewProps {
  vectorStoreId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editVectorStore: boolean;
}

interface EditFormValues {
  vector_store_id: string;
  vector_store_name?: string;
  vector_store_description?: string;
  custom_llm_provider: string;
  litellm_credential_name?: string | null;
}

function LabelWithTooltip({
  children,
  tooltip,
  required,
  htmlFor,
}: {
  children: React.ReactNode;
  tooltip: string;
  required?: boolean;
  htmlFor?: string;
}) {
  return (
    <Label htmlFor={htmlFor} className="flex items-center gap-1">
      <span>
        {children}
        {required && <span className="text-destructive"> *</span>}
      </span>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="h-3 w-3 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </Label>
  );
}

const VectorStoreInfoView: React.FC<VectorStoreInfoViewProps> = ({
  vectorStoreId,
  onClose,
  accessToken,
  is_admin,
  editVectorStore,
}) => {
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStore | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editVectorStore);
  const [metadataString, setMetadataString] = useState<string>("{}");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);

  const {
    register,
    handleSubmit,
    control,
    reset,
    formState: { errors },
  } = useForm<EditFormValues>({
    defaultValues: {
      vector_store_id: "",
      vector_store_name: "",
      vector_store_description: "",
      custom_llm_provider: "bedrock",
      litellm_credential_name: null,
    },
  });

  const fetchVectorStoreDetails = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreInfoCall(accessToken, vectorStoreId);
      if (response && response.vector_store) {
        setVectorStoreDetails(response.vector_store);

        if (response.vector_store.vector_store_metadata) {
          const metadata =
            typeof response.vector_store.vector_store_metadata === "string"
              ? JSON.parse(response.vector_store.vector_store_metadata)
              : response.vector_store.vector_store_metadata;
          setMetadataString(JSON.stringify(metadata, null, 2));
        }

        if (editVectorStore) {
          reset({
            vector_store_id: response.vector_store.vector_store_id,
            custom_llm_provider: response.vector_store.custom_llm_provider,
            vector_store_name: response.vector_store.vector_store_name,
            vector_store_description: response.vector_store.vector_store_description,
            litellm_credential_name: null,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching vector store details:", error);
      NotificationsManager.fromBackend("Error fetching vector store details: " + error);
    }
  }, [accessToken, editVectorStore, reset, vectorStoreId]);

  const fetchCredentials = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchVectorStoreDetails();
    fetchCredentials();
  }, [fetchVectorStoreDetails, fetchCredentials]);

  const handleSave = async (values: EditFormValues) => {
    if (!accessToken) return;
    try {
      let metadata: Record<string, any> = {};
      try {
        metadata = metadataString ? JSON.parse(metadataString) : {};
      } catch {
        NotificationsManager.fromBackend("Invalid JSON in metadata field");
        return;
      }

      const updateData = {
        vector_store_id: values.vector_store_id,
        custom_llm_provider: values.custom_llm_provider,
        vector_store_name: values.vector_store_name,
        vector_store_description: values.vector_store_description,
        vector_store_metadata: metadata,
      };

      await vectorStoreUpdateCall(accessToken, updateData);
      NotificationsManager.success("Vector store updated successfully");
      setIsEditing(false);
      fetchVectorStoreDetails();
    } catch (error) {
      console.error("Error updating vector store:", error);
      NotificationsManager.fromBackend("Error updating vector store: " + error);
    }
  };

  if (!vectorStoreDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" className="mb-4" onClick={onClose}>
            <ArrowLeft className="h-4 w-4" />
            Back to Vector Stores
          </Button>
          <h1 className="text-2xl font-semibold">
            Vector Store ID: {vectorStoreDetails.vector_store_id}
          </h1>
          <p className="text-muted-foreground">
            {vectorStoreDetails.vector_store_description || "No description"}
          </p>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>
        )}
      </div>

      <Tabs defaultValue="details">
        <TabsList className="mb-6">
          <TabsTrigger value="details">Details</TabsTrigger>
          <TabsTrigger value="test">Test Vector Store</TabsTrigger>
        </TabsList>

        <TabsContent value="details">
          {isEditing ? (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">Edit Vector Store</h2>
              </div>
              <Card className="p-4">
                <form onSubmit={handleSubmit(handleSave)} className="space-y-4">
                  <div className="space-y-1">
                    <Label htmlFor="vs-edit-id">
                      Vector Store ID <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      id="vs-edit-id"
                      disabled
                      {...register("vector_store_id", {
                        required: "Please input a vector store ID",
                      })}
                    />
                    {errors.vector_store_id && (
                      <p className="text-xs text-destructive">
                        {errors.vector_store_id.message as string}
                      </p>
                    )}
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="vs-edit-name">Vector Store Name</Label>
                    <Input id="vs-edit-name" {...register("vector_store_name")} />
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="vs-edit-description">Description</Label>
                    <Textarea
                      id="vs-edit-description"
                      rows={4}
                      {...register("vector_store_description")}
                    />
                  </div>

                  <div className="space-y-1">
                    <LabelWithTooltip
                      htmlFor="vs-edit-provider"
                      tooltip="Select the provider for this vector store"
                      required
                    >
                      Provider
                    </LabelWithTooltip>
                    <Controller
                      control={control}
                      name="custom_llm_provider"
                      rules={{ required: "Please select a provider" }}
                      render={({ field }) => (
                        <Select value={field.value} onValueChange={field.onChange}>
                          <SelectTrigger id="vs-edit-provider" className="w-full">
                            <SelectValue placeholder="Select a provider" />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.entries(Providers)
                              .filter(([providerEnum]) => providerEnum === "Bedrock")
                              .map(([providerEnum, providerDisplayName]) => (
                                <SelectItem
                                  key={providerEnum}
                                  value={provider_map[providerEnum]}
                                >
                                  <div className="flex items-center space-x-2">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={providerLogoMap[providerDisplayName]}
                                      alt={`${providerEnum} logo`}
                                      className="w-5 h-5"
                                      onError={(e) => {
                                        const target = e.target as HTMLImageElement;
                                        const parent = target.parentElement;
                                        if (parent) {
                                          const fallbackDiv = document.createElement("div");
                                          fallbackDiv.className =
                                            "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                          fallbackDiv.textContent =
                                            providerDisplayName.charAt(0);
                                          parent.replaceChild(fallbackDiv, target);
                                        }
                                      }}
                                    />
                                    <span>{providerDisplayName}</span>
                                  </div>
                                </SelectItem>
                              ))}
                          </SelectContent>
                        </Select>
                      )}
                    />
                    {errors.custom_llm_provider && (
                      <p className="text-xs text-destructive">
                        {errors.custom_llm_provider.message as string}
                      </p>
                    )}
                  </div>

                  <div className="mb-4">
                    <p className="text-sm text-muted-foreground mb-2">
                      Either select existing credentials OR enter provider credentials below
                    </p>
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="vs-edit-credential">Existing Credentials</Label>
                    <Controller
                      control={control}
                      name="litellm_credential_name"
                      render={({ field }) => (
                        <Select
                          value={field.value ?? "__none__"}
                          onValueChange={(v) =>
                            field.onChange(v === "__none__" ? null : v)
                          }
                        >
                          <SelectTrigger id="vs-edit-credential" className="w-full">
                            <SelectValue placeholder="Select or search for existing credentials" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__none__">None</SelectItem>
                            {credentials.map((credential) => (
                              <SelectItem
                                key={credential.credential_name}
                                value={credential.credential_name}
                              >
                                {credential.credential_name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                    />
                  </div>

                  <div className="flex items-center my-4">
                    <div className="flex-grow border-t border-border"></div>
                    <span className="px-4 text-muted-foreground text-sm">OR</span>
                    <div className="flex-grow border-t border-border"></div>
                  </div>

                  <div className="space-y-1">
                    <LabelWithTooltip
                      htmlFor="vs-edit-metadata"
                      tooltip="JSON metadata for the vector store"
                    >
                      Metadata
                    </LabelWithTooltip>
                    <Textarea
                      id="vs-edit-metadata"
                      rows={4}
                      value={metadataString}
                      onChange={(e) => setMetadataString(e.target.value)}
                      placeholder='{"key": "value"}'
                    />
                  </div>

                  <div className="flex justify-end space-x-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setIsEditing(false)}
                    >
                      Cancel
                    </Button>
                    <Button type="submit">Save Changes</Button>
                  </div>
                </form>
              </Card>
            </div>
          ) : (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">Vector Store Details</h2>
                {is_admin && (
                  <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>
                )}
              </div>
              <Card className="p-4">
                <div className="space-y-4">
                  <div>
                    <p className="font-medium">ID</p>
                    <p>{vectorStoreDetails.vector_store_id}</p>
                  </div>
                  <div>
                    <p className="font-medium">Name</p>
                    <p>{vectorStoreDetails.vector_store_name || "-"}</p>
                  </div>
                  <div>
                    <p className="font-medium">Description</p>
                    <p>{vectorStoreDetails.vector_store_description || "-"}</p>
                  </div>
                  <div>
                    <p className="font-medium">Provider</p>
                    <div className="flex items-center space-x-2 mt-1">
                      {(() => {
                        const provider =
                          vectorStoreDetails.custom_llm_provider || "bedrock";
                        const { displayName, logo } = (() => {
                          const enumKey = Object.keys(provider_map).find(
                            (key) =>
                              provider_map[key].toLowerCase() === provider.toLowerCase(),
                          );

                          if (!enumKey) {
                            return { displayName: provider, logo: "" };
                          }

                          const displayName = Providers[enumKey as keyof typeof Providers];
                          const logo = providerLogoMap[displayName];

                          return { displayName, logo };
                        })();

                        return (
                          <>
                            {logo && (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img
                                src={logo}
                                alt={`${displayName} logo`}
                                className="w-5 h-5"
                                onError={(e) => {
                                  const target = e.target as HTMLImageElement;
                                  const parent = target.parentElement;
                                  if (parent) {
                                    const fallbackDiv = document.createElement("div");
                                    fallbackDiv.className =
                                      "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                    fallbackDiv.textContent = displayName.charAt(0);
                                    parent.replaceChild(fallbackDiv, target);
                                  }
                                }}
                              />
                            )}
                            <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                              {displayName}
                            </Badge>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                  <div>
                    <p className="font-medium">Metadata</p>
                    <div className="bg-muted p-3 rounded mt-2 font-mono text-xs overflow-auto max-h-48">
                      <pre>{metadataString}</pre>
                    </div>
                  </div>
                  <div>
                    <p className="font-medium">Created</p>
                    <p>
                      {vectorStoreDetails.created_at
                        ? new Date(vectorStoreDetails.created_at).toLocaleString()
                        : "-"}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Last Updated</p>
                    <p>
                      {vectorStoreDetails.updated_at
                        ? new Date(vectorStoreDetails.updated_at).toLocaleString()
                        : "-"}
                    </p>
                  </div>
                </div>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="test">
          <VectorStoreTester
            vectorStoreId={vectorStoreDetails.vector_store_id}
            accessToken={accessToken || ""}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default VectorStoreInfoView;
