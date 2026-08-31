import { useQueryClient } from "@tanstack/react-query";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { ChevronRight, CircleHelp, Info, UserPlus } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import TeamDropdown from "./common_components/team_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import { toast } from "@/lib/toast";
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
  metadata?: string;
  send_invite_email: boolean;
  models?: string[];
}

const EMBEDDED_DEFAULTS: CreateUserFormValues = {
  user_email: undefined,
  user_role: "internal_user_viewer",
  team_id: undefined,
  metadata: undefined,
  send_invite_email: true,
};

const STANDALONE_DEFAULTS: CreateUserFormValues = {
  user_email: undefined,
  user_role: "internal_user_viewer",
  team_id: undefined,
  organization_ids: undefined,
  metadata: undefined,
  send_invite_email: true,
};

// antd reported only mounted fields, so a collapsed Personal Key Creation
// section left `models` out of the request entirely even after the admin had
// picked some and closed it again.
const withoutCollapsedModels = (values: CreateUserFormValues, isPersonalKeyOpen: boolean): CreateUserFormValues => {
  if (isPersonalKeyOpen) {
    return values;
  }
  const { models, ...rest } = values;
  return rest;
};

const buildCreatePayload = (values: CreateUserFormValues): CreateUserFormValues & { organizations?: string[] } => {
  const withModels =
    (!values.models || values.models.length === 0) && values.user_role !== "proxy_admin"
      ? { ...values, models: ["no-default-models"] }
      : values;
  if (!withModels.organization_ids) {
    return withModels;
  }
  const { organization_ids, ...rest } = withModels;
  return { ...rest, organizations: organization_ids };
};

const SSO_INVITE_LIFETIME_MS = 7 * 24 * 60 * 60 * 1000;

const buildSsoInvitationLink = (userId: string, createdBy: string): InvitationLink => {
  const now = new Date();
  return {
    id: generateUUID(),
    user_id: userId,
    is_accepted: false,
    accepted_at: null,
    expires_at: new Date(now.getTime() + SSO_INVITE_LIFETIME_MS),
    created_at: now,
    created_by: createdBy,
    updated_at: now,
    updated_by: createdBy,
    has_user_setup_sso: true,
  };
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const EmailInvitationsNotice: React.FC = () => (
  <Alert variant="info" className="mb-4">
    <Info />
    <AlertTitle>Email invitations</AlertTitle>
    <AlertDescription>
      New users receive an email invite only when an email integration (SMTP, Resend, or SendGrid) is configured.{" "}
      <a href="https://docs.litellm.ai/docs/proxy/email" target="_blank" rel="noreferrer">
        Learn how to set up email notifications
      </a>
    </AlertDescription>
  </Alert>
);

export const CreateUserButton: React.FC<CreateuserProps> = ({
  userID,
  accessToken,
  possibleUIRoles,
  onUserCreated,
  isEmbedded = false,
}) => {
  const queryClient = useQueryClient();
  const [uiSettings, setUISettings] = useState<UISettings | null>(null);
  const defaultValues = isEmbedded ? EMBEDDED_DEFAULTS : STANDALONE_DEFAULTS;
  const form = useForm<CreateUserFormValues>({ defaultValues });
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiuser, setApiuser] = useState<boolean>(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isPersonalKeyOpen, setIsPersonalKeyOpen] = useState(false);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const { data: organizations = [] } = useOrganizations();
  const organizationOptions = organizations.map((org) => ({
    label: `${org.organization_alias} (${org.organization_id})`,
    value: org.organization_id ?? "",
  }));

  useEffect(() => {
    const fetchData = async () => {
      try {
        const userRole = "any";
        const modelDataResponse = await modelAvailableCall(accessToken, userID, userRole);
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
  }, []);

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiuser(false);
    form.reset(defaultValues);
  };

  const handleCreate = async (formValues: CreateUserFormValues) => {
    try {
      toast.info("Making API Call");
      if (!isEmbedded) {
        setIsModalVisible(true);
      }
      const payload = buildCreatePayload(withoutCollapsedModels(formValues, isPersonalKeyOpen));
      const response = await userCreateCall(accessToken, null, payload);
      await queryClient.invalidateQueries({ queryKey: ["userList"] });
      setApiuser(true);
      const user_id = response.data?.user_id || response.user_id;

      if (onUserCreated && isEmbedded) {
        onUserCreated(user_id);
        form.reset(defaultValues);
        return;
      }

      if (!uiSettings?.SSO_ENABLED) {
        invitationCreateCall(accessToken, user_id).then((data) => {
          data.has_user_setup_sso = false;
          setInvitationLinkData(data);
          setIsInvitationLinkModalVisible(true);
        });
      } else {
        setInvitationLinkData(buildSsoInvitationLink(user_id, userID));
        setIsInvitationLinkModalVisible(true);
      }

      toast.success("API user Created");
      form.reset(defaultValues);
      localStorage.removeItem("userData" + userID);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error?.message || "Error creating the user";
      toast.fromError(errorMessage);
      console.error("Error creating the user:", error);
    }
  };

  const roleOptions = Object.entries(possibleUIRoles ?? {}).map(([role, { ui_label, description }]) => ({
    value: role,
    label: ui_label,
    description,
  }));

  const userEmailField = (
    <FormField control={form.control} name="user_email" label="User Email">
      {({ ref, value, ...control }) => <Input {...control} ref={ref} value={value ?? ""} />}
    </FormField>
  );

  const teamField = (
    <FormField
      control={form.control}
      name="team_id"
      label="Team"
      description="If selected, user will be added as a 'user' role to the team."
    >
      {({ id, value, onChange }) => <TeamDropdown id={id} value={value} onChange={onChange} />}
    </FormField>
  );

  const metadataField = (
    <FormField control={form.control} name="metadata" label="Metadata">
      {({ ref, value, ...control }) => (
        <Textarea {...control} ref={ref} value={value ?? ""} rows={4} placeholder="Enter metadata as JSON" />
      )}
    </FormField>
  );

  const sendInviteEmailField = (
    <FormField control={form.control} name="send_invite_email" label="Send invitation email">
      {({ id, value, onChange, onBlur }) => (
        <Checkbox id={id} checked={value} onCheckedChange={onChange} onBlur={onBlur} />
      )}
    </FormField>
  );

  const roleField = (label: React.ReactNode) => (
    <FormField control={form.control} name="user_role" label={label}>
      {({ id, value, onChange }) => (
        <Select
          items={roleOptions}
          value={value === undefined || value === "" ? null : value}
          onValueChange={(selected: string | null) => onChange(selected ?? undefined)}
        >
          <SelectTrigger id={id} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {roleOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                <span>{option.label}</span>
                <span className="ml-2 text-xs text-muted-foreground">{option.description}</span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </FormField>
  );

  // Modify the return statement to handle embedded mode
  if (isEmbedded) {
    return (
      <TooltipProvider>
        <form onSubmit={form.handleSubmit(handleCreate)}>
          <EmailInvitationsNotice />
          <FieldGroup>
            {userEmailField}
            {roleField("User Role")}
            {teamField}
            {metadataField}
            {sendInviteEmailField}
          </FieldGroup>
          <div className="mt-4 text-right">
            <Button type="submit">Create User</Button>
          </div>
        </form>
      </TooltipProvider>
    );
  }

  // Original return for standalone mode
  return (
    <>
      <Button type="button" onClick={() => setIsModalVisible(true)}>
        + Invite User
      </Button>
      <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>Invite User</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <p className="mb-1 text-sm text-foreground">Create a User who can own keys</p>
            <EmailInvitationsNotice />
          </div>
          <TooltipProvider>
            <form onSubmit={form.handleSubmit(handleCreate)}>
              <FieldGroup>
                {userEmailField}
                {roleField(
                  labelWithHint(
                    "Global Proxy Role",
                    "This role is independent of any team/org specific roles. Configure Team / Organization Admins in the Settings",
                  ),
                )}
                {teamField}

                <FormField
                  control={form.control}
                  name="organization_ids"
                  label="Organization"
                  description="The user will be added to the selected organization(s)."
                >
                  {({ id, value, onChange }) => (
                    <Select
                      multiple
                      items={organizationOptions}
                      value={value ?? []}
                      onValueChange={(selected: string[]) => onChange(selected.length === 0 ? undefined : selected)}
                    >
                      <SelectTrigger id={id} className="w-full">
                        <SelectValue placeholder="Select Organization">
                          {(selected: string[]) =>
                            selected.length === 0
                              ? "Select Organization"
                              : organizationOptions
                                  .filter((option) => selected.includes(option.value))
                                  .map((option) => option.label)
                                  .join(", ")
                          }
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {organizationOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </FormField>

                {metadataField}
                {sendInviteEmailField}

                <Collapsible open={isPersonalKeyOpen} onOpenChange={setIsPersonalKeyOpen}>
                  <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-md border border-border px-3 py-2 text-left text-sm font-semibold text-foreground">
                    <ChevronRight
                      className={`size-4 transition-transform ${isPersonalKeyOpen ? "rotate-90" : ""}`}
                      aria-hidden
                    />
                    Personal Key Creation
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-4">
                    <FormField
                      control={form.control}
                      name="models"
                      label={labelWithHint("Models", "Models user has access to, outside of team scope.")}
                      description="Models user has access to, outside of team scope."
                    >
                      {({ value, onChange }) => (
                        <MultiSelect
                          options={[
                            { label: "All Proxy Models", value: "all-proxy-models" },
                            { label: "No Default Models", value: "no-default-models" },
                            ...userModels.map((model) => ({ label: getModelDisplayName(model), value: model })),
                          ]}
                          value={value ?? []}
                          onValueChange={onChange}
                          placeholder="Select models"
                        />
                      )}
                    </FormField>
                  </CollapsibleContent>
                </Collapsible>
              </FieldGroup>

              <div className="mt-4 text-right">
                <Button type="submit">
                  <UserPlus />
                  Invite User
                </Button>
              </div>
            </form>
          </TooltipProvider>
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
    </>
  );
};
