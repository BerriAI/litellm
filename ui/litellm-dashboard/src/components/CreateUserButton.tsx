import React, { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Controller, useForm } from "react-hook-form";
import { Info, UserPlus, X } from "lucide-react";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
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
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import BulkCreateUsers from "./bulk_create_users_button";
import TeamDropdown from "./common_components/team_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NotificationsManager from "./molecules/notifications_manager";
import {
  getProxyBaseUrl,
  getProxyUISettings,
  invitationCreateCall,
  modelAvailableCall,
  userCreateCall,
} from "./networking";
import OnboardingModal, { InvitationLink } from "./onboarding_link";

// Helper function to generate UUID compatible across all environments
const generateUUID = (): string => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback UUID generation for environments without crypto.randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

interface CreateuserProps {
  userID: string;
  accessToken: string;
  teams: any[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onUserCreated?: (userId: string) => void;
  isEmbedded?: boolean;
}

// Define an interface for the UI settings
interface UISettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
}

interface CreateUserFormValues {
  user_email?: string;
  user_role: string;
  team_id?: string;
  organization_ids?: string[];
  models?: string[];
  metadata?: string;
}

const defaultFormValues: CreateUserFormValues = {
  user_email: "",
  user_role: "internal_user_viewer",
  team_id: undefined,
  organization_ids: [],
  models: [],
  metadata: "",
};

/**
 * shadcn Select + badge chip multi-select. Mirrors the pattern from
 * AccessGroupBaseForm. Used for org IDs and personal model selection.
 */
function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText,
  ariaLabel,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText: string;
  ariaLabel?: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger aria-label={ariaLabel ?? placeholder}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

export const CreateUserButton: React.FC<CreateuserProps> = ({
  userID,
  accessToken,
  teams,
  possibleUIRoles,
  onUserCreated,
  isEmbedded = false,
}) => {
  const queryClient = useQueryClient();
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiuser, setApiuser] = useState<boolean>(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] =
    useState(false);
  const [invitationLinkData, setInvitationLinkData] =
    useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const { data: organizations = [] } = useOrganizations();

  const {
    control,
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<CreateUserFormValues>({ defaultValues: defaultFormValues });

  // Derive teams from the user's organizations, falling back to the teams prop
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const availableTeams = useMemo(() => {
    const orgTeams = organizations.flatMap((org) => org.teams || []);
    if (orgTeams.length > 0) return orgTeams;
    return teams || [];
  }, [organizations, teams]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const userRole = "any";
        const modelDataResponse = await modelAvailableCall(
          accessToken,
          userID,
          userRole,
        );
        const availableModels = [];
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          availableModels.push(model.id);
        }
        setUserModels(availableModels);
        const uiSettingsResponse = await getProxyUISettings(accessToken);
        setUISettings(uiSettingsResponse);
      } catch (error) {
        console.error("Error fetching model data:", error);
      }
    };

    setBaseUrl(getProxyBaseUrl());
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiuser(false);
    reset(defaultFormValues);
  };

  const handleCreate = async (formValues: CreateUserFormValues) => {
    try {
      NotificationsManager.info("Making API Call");
      if (!isEmbedded) {
        setIsModalVisible(true);
      }

      const payload: Record<string, any> = {
        user_email: formValues.user_email,
        user_role: formValues.user_role,
        team_id: formValues.team_id,
        models: formValues.models,
        metadata: formValues.metadata,
      };

      if (
        (!payload.models || payload.models.length === 0) &&
        payload.user_role !== "proxy_admin"
      ) {
        payload.models = ["no-default-models"];
      }
      if (formValues.organization_ids && formValues.organization_ids.length > 0) {
        payload.organizations = formValues.organization_ids;
      }
      const response = await userCreateCall(accessToken, null, payload);
      await queryClient.invalidateQueries({ queryKey: ["userList"] });
      setApiuser(true);
      const user_id = response.data?.user_id || response.user_id;

      if (onUserCreated && isEmbedded) {
        onUserCreated(user_id);
        reset(defaultFormValues);
        return;
      }

      if (!uiSettings?.SSO_ENABLED) {
        invitationCreateCall(accessToken, user_id).then((data) => {
          data.has_user_setup_sso = false;
          setInvitationLinkData(data);
          setIsInvitationLinkModalVisible(true);
        });
      } else {
        // create an InvitationLink Object for this user for the SSO flow
        // for SSO the invite link is the proxy base url since the User just needs to login
        const invitationLink: InvitationLink = {
          id: generateUUID(),
          user_id: user_id,
          is_accepted: false,
          accepted_at: null,
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
          created_at: new Date(),
          created_by: userID,
          updated_at: new Date(),
          updated_by: userID,
          has_user_setup_sso: true,
        };
        setInvitationLinkData(invitationLink);
        setIsInvitationLinkModalVisible(true);
      }

      NotificationsManager.success("API user Created");
      reset(defaultFormValues);
      localStorage.removeItem("userData" + userID);
    } catch (error: any) {
      const errorMessage =
        error.response?.data?.detail || error?.message || "Error creating the user";
      NotificationsManager.fromBackend(errorMessage);
      console.error("Error creating the user:", error);
    }
  };

  const orgOptions = organizations.map((org) => ({
    label: `${org.organization_alias} (${org.organization_id})`,
    value: org.organization_id,
  }));

  const modelOptions = useMemo(() => {
    const base = [
      { label: "All Proxy Models", value: "all-proxy-models" },
      { label: "No Default Models", value: "no-default-models" },
    ];
    const extras = userModels.map((model) => ({
      label: getModelDisplayName(model),
      value: model,
    }));
    return [...base, ...extras];
  }, [userModels]);

  // Modify the return statement to handle embedded mode
  if (isEmbedded) {
    return (
      <form onSubmit={handleSubmit(handleCreate)} className="space-y-4">
        <Alert>
          <AlertTitle>Email invitations</AlertTitle>
          <AlertDescription>
            New users receive an email invite only when an email integration
            (SMTP, Resend, or SendGrid) is configured.{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/email"
              target="_blank"
              rel="noreferrer"
              className="text-primary underline"
            >
              Learn how to set up email notifications
            </a>
          </AlertDescription>
        </Alert>

        <div className="space-y-2">
          <Label htmlFor="embedded_user_email">User Email</Label>
          <Input id="embedded_user_email" {...register("user_email")} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="embedded_user_role">User Role</Label>
          <Controller
            control={control}
            name="user_role"
            render={({ field }) => (
              <Select
                value={field.value ?? ""}
                onValueChange={(v) => field.onChange(v)}
              >
                <SelectTrigger id="embedded_user_role">
                  <SelectValue placeholder="Select a role" />
                </SelectTrigger>
                <SelectContent>
                  {possibleUIRoles &&
                    Object.entries(possibleUIRoles).map(
                      ([role, { ui_label, description }]) => (
                        <SelectItem key={role} value={role} title={ui_label}>
                          <div className="flex">
                            {ui_label}
                            <span className="ml-2 text-muted-foreground text-xs">
                              {description}
                            </span>
                          </div>
                        </SelectItem>
                      ),
                    )}
                </SelectContent>
              </Select>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Team</Label>
          <Controller
            control={control}
            name="team_id"
            render={({ field }) => (
              <TeamDropdown
                value={field.value ?? ""}
                onChange={(v) => field.onChange(v)}
              />
            )}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="embedded_metadata">Metadata</Label>
          <Textarea
            id="embedded_metadata"
            rows={4}
            placeholder="Enter metadata as JSON"
            {...register("metadata")}
          />
        </div>

        <div className="flex justify-end">
          <Button type="submit">Create User</Button>
        </div>
      </form>
    );
  }

  // Original return for standalone mode
  return (
    <div className="flex gap-2">
      <Button onClick={() => setIsModalVisible(true)}>+ Invite User</Button>
      <BulkCreateUsers
        accessToken={accessToken}
        teams={teams}
        possibleUIRoles={possibleUIRoles}
      />

      <Dialog
        open={isModalVisible}
        onOpenChange={(o) => (!o ? handleCancel() : undefined)}
      >
        <DialogContent className="max-w-[800px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Invite User</DialogTitle>
            <DialogDescription>Create a User who can own keys</DialogDescription>
          </DialogHeader>

          <Alert>
            <AlertTitle>Email invitations</AlertTitle>
            <AlertDescription>
              New users receive an email invite only when an email integration
              (SMTP, Resend, or SendGrid) is configured.{" "}
              <a
                href="https://docs.litellm.ai/docs/proxy/email"
                target="_blank"
                rel="noreferrer"
                className="text-primary underline"
              >
                Learn how to set up email notifications
              </a>
            </AlertDescription>
          </Alert>

          <form
            onSubmit={handleSubmit(handleCreate)}
            className="space-y-4 mt-4"
          >
            <div className="space-y-2">
              <Label htmlFor="user_email">User Email</Label>
              <Input id="user_email" {...register("user_email")} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="user_role" className="flex items-center">
                Global Proxy Role
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      This role is independent of any team/org specific roles.
                      Configure Team / Organization Admins in the Settings
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </Label>
              <Controller
                control={control}
                name="user_role"
                render={({ field }) => (
                  <Select
                    value={field.value ?? ""}
                    onValueChange={(v) => field.onChange(v)}
                  >
                    <SelectTrigger id="user_role">
                      <SelectValue placeholder="Select a role" />
                    </SelectTrigger>
                    <SelectContent>
                      {possibleUIRoles &&
                        Object.entries(possibleUIRoles).map(
                          ([role, { ui_label, description }]) => (
                            <SelectItem
                              key={role}
                              value={role}
                              title={ui_label}
                            >
                              <span className="font-medium">{ui_label}</span>
                              <span className="ml-2 text-muted-foreground text-xs">
                                {" - "}
                                {description}
                              </span>
                            </SelectItem>
                          ),
                        )}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>

            <div className="space-y-2">
              <Label>Team</Label>
              <Controller
                control={control}
                name="team_id"
                render={({ field }) => (
                  <TeamDropdown
                    value={field.value ?? ""}
                    onChange={(v) => field.onChange(v)}
                  />
                )}
              />
              <p className="text-xs text-muted-foreground">
                If selected, user will be added as a &apos;user&apos; role to
                the team.
              </p>
            </div>

            <div className="space-y-2">
              <Label>Organization</Label>
              <Controller
                control={control}
                name="organization_ids"
                render={({ field }) => (
                  <MultiSelect
                    value={field.value ?? []}
                    onChange={field.onChange}
                    options={orgOptions}
                    placeholder="Select Organization"
                    emptyText="No organizations available"
                    ariaLabel="Organization"
                  />
                )}
              />
              <p className="text-xs text-muted-foreground">
                The user will be added to the selected organization(s).
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="metadata">Metadata</Label>
              <Textarea
                id="metadata"
                rows={4}
                placeholder="Enter metadata as JSON"
                {...register("metadata")}
              />
            </div>

            <Accordion type="single" collapsible>
              <AccordionItem value="personal-key-creation">
                <AccordionTrigger>
                  <span className="font-medium">Personal Key Creation</span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-2">
                    <Label className="flex items-center">
                      Models
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            Models user has access to, outside of team scope.
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </Label>
                    <Controller
                      control={control}
                      name="models"
                      render={({ field }) => (
                        <MultiSelect
                          value={field.value ?? []}
                          onChange={field.onChange}
                          options={modelOptions}
                          placeholder="Select models"
                          emptyText="No models available"
                        />
                      )}
                    />
                    <p className="text-xs text-muted-foreground">
                      Models user has access to, outside of team scope.
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            {errors.user_role && (
              <p className="text-sm text-destructive">
                {errors.user_role.message as string}
              </p>
            )}

            <DialogFooter>
              <Button type="submit">
                <UserPlus className="h-4 w-4 mr-1" />
                Invite User
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {apiuser && (
        <OnboardingModal
          isInvitationLinkModalVisible={isInvitationLinkModalVisible}
          setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
          baseUrl={baseUrl || ""}
          invitationLinkData={invitationLinkData}
        />
      )}
    </div>
  );
};
