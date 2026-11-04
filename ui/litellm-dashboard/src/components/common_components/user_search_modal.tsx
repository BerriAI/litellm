import { useRef, useState } from "react";
import type { TFunction } from "i18next";
import { Info, UserPlus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { useForm } from "react-hook-form";
import { userFilterUICall } from "@/components/networking";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { PaginatedSearchSelect } from "@/components/shared/PaginatedSearchSelect";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User | null;
}

interface Role {
  label: string;
  value: string;
  description: string;
}

interface FormValues {
  user_email: string | null | undefined;
  user_id: string | null | undefined;
  role: string;
}

interface UserSearchModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSubmit: (values: FormValues) => void | Promise<void>;
  accessToken: string | null;
  title?: string;
  roles?: Role[];
  defaultRole?: string;
  teamId?: string;
}

const defaultRoles = (t: TFunction): Role[] => [
  { label: "admin", value: "admin", description: t("userSearchModal.defaultRoles.adminDescription") },
  { label: "user", value: "user", description: t("userSearchModal.defaultRoles.userDescription") },
];

const UserSearchModal: React.FC<UserSearchModalProps> = ({
  isVisible,
  onCancel,
  onSubmit,
  accessToken,
  title,
  roles,
  defaultRole = "user",
  teamId,
}) => {
  const { t } = useTranslation();
  const roleOptions = roles ?? defaultRoles(t);
  const emptyValues: FormValues = { user_email: undefined, user_id: undefined, role: defaultRole };
  const form = useForm<FormValues>({ defaultValues: emptyValues });
  const selectedUserId = form.watch("user_id");
  const selectedUserEmail = form.watch("user_email");
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<"user_email" | "user_id">("user_email");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const latestSearchRef = useRef(0);

  const fetchUsers = async (searchText: string, fieldName: "user_email" | "user_id"): Promise<void> => {
    const searchId = latestSearchRef.current + 1;
    latestSearchRef.current = searchId;
    const isLatestSearch = (): boolean => searchId === latestSearchRef.current;

    if (!searchText) {
      setUserOptions([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append(fieldName, searchText);
      if (teamId) {
        params.append("team_id", teamId);
      }
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);
      if (!isLatestSearch()) return;

      const data: User[] = response;
      const options: UserOption[] = data.map((user) => ({
        label: fieldName === "user_email" ? `${user.user_email}` : `${user.user_id}`,
        value: fieldName === "user_email" ? user.user_email : user.user_id,
        user,
      }));
      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
    } finally {
      if (isLatestSearch()) setLoading(false);
    }
  };

  const handleSearch = (value: string, fieldName: "user_email" | "user_id"): void => {
    setSelectedField(fieldName);
    void fetchUsers(value, fieldName);
  };

  const handleSelect = (option: UserOption | null): void => {
    if (option?.user == null) return;
    form.setValue("user_email", option.user.user_email);
    form.setValue("user_id", option.user.user_id);
  };

  const handleSubmit = async (values: FormValues): Promise<void> => {
    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = (): void => {
    form.reset(emptyValues);
    setUserOptions([]);
    onCancel();
  };

  const swallowEnter = (event: React.KeyboardEvent): void => {
    if (event.key === "Enter") event.preventDefault();
  };

  const renderUserSearch = (
    fieldName: "user_email" | "user_id",
    placeholder: string,
    controlProps: {
      id: string;
      value: string | null | undefined;
      onChange: (value: string | null | undefined) => void;
    },
    testId?: string,
  ) => {
    const items = selectedField === fieldName ? userOptions : [];
    const handleValueChange = (value: string | null) => {
      if (value === null) {
        form.setValue("user_email", null);
        form.setValue("user_id", null);
        return;
      }
      controlProps.onChange(value);
      handleSelect(items.find((option) => option.value === value) ?? null);
    };
    return (
      <div data-testid={testId} onKeyDown={swallowEnter}>
        <PaginatedSearchSelect
          options={items}
          value={controlProps.value}
          onValueChange={handleValueChange}
          onSearchChange={(query: string) => handleSearch(query, fieldName)}
          autoHighlight="always"
          isLoading={loading}
          placeholder={placeholder}
          emptyText={t("userSearchModal.search.noResults")}
          loadingText={t("userSearchModal.search.loading")}
          inputId={controlProps.id}
        />
      </div>
    );
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleClose()} disablePointerDismissal={isSubmitting}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{title ?? t("userSearchModal.defaultTitle")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleSubmit)} noValidate>
            <Alert variant="info" className="mb-4" data-testid="member-existing-users-notice">
              <Info />
              <AlertTitle>{t("userSearchModal.existingUsersNotice")}</AlertTitle>
            </Alert>

            <FieldGroup>
              <FormField control={form.control} name="user_email" label={t("userSearchModal.fields.email")}>
                {({ id, value, onChange }) =>
                  renderUserSearch(
                    "user_email",
                    t("userSearchModal.fields.emailPlaceholder"),
                    { id, value, onChange },
                    "member-email-search",
                  )
                }
              </FormField>

              <div className="text-center">{t("userSearchModal.fields.or")}</div>

              <FormField control={form.control} name="user_id" label={t("userSearchModal.fields.userId")}>
                {({ id, value, onChange }) =>
                  renderUserSearch("user_id", t("userSearchModal.fields.userIdPlaceholder"), { id, value, onChange })
                }
              </FormField>

              <FormField control={form.control} name="role" label={t("userSearchModal.fields.role")}>
                {({ id, value, onChange }) => (
                  <Select items={roleOptions} value={value} onValueChange={(next) => onChange(next as string)}>
                    <SelectTrigger id={id}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {roleOptions.map((role) => (
                        <SelectItem key={role.value} value={role.value}>
                          <Tooltip>
                            <TooltipTrigger
                              render={
                                <span>
                                  <span className="font-medium">{role.label}</span>
                                  <span className="ml-2 text-sm text-muted-foreground">- {role.description}</span>
                                </span>
                              }
                            />
                            <TooltipContent>{role.description}</TooltipContent>
                          </Tooltip>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </FormField>
            </FieldGroup>

            <div className="mt-4 text-right">
              <Button type="submit" disabled={isSubmitting || (!selectedUserId && !selectedUserEmail)}>
                {isSubmitting ? <UiLoadingSpinner className="size-4" /> : <UserPlus />}
                {isSubmitting ? t("userSearchModal.submit.adding") : t("userSearchModal.submit.add")}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default UserSearchModal;
