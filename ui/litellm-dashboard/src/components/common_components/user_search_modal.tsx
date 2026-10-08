import { useRef, useState } from "react";
import { Info, UserPlus } from "lucide-react";
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

type SearchField = "user_email" | "user_id";

interface UserOption {
  label: string;
  value: string;
  users: User[];
}

const groupUsersByField = (users: User[], fieldName: SearchField): UserOption[] =>
  [...new Set(users.map((user) => `${user[fieldName]}`))].map((value) => ({
    label: value,
    value,
    users: users.filter((user) => `${user[fieldName]}` === value),
  }));

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

const UserSearchModal: React.FC<UserSearchModalProps> = ({
  isVisible,
  onCancel,
  onSubmit,
  accessToken,
  title = "Add Team Member",
  roles = [
    {
      label: "admin",
      value: "admin",
      description: "Admin role. Can create team keys, add members, and manage settings.",
    },
    { label: "user", value: "user", description: "User role. Can view team info, but not manage it." },
  ],
  defaultRole = "user",
  teamId,
}) => {
  const emptyValues: FormValues = { user_email: undefined, user_id: undefined, role: defaultRole };
  const form = useForm<FormValues>({ defaultValues: emptyValues });
  const selectedUserId = form.watch("user_id");
  const selectedUserEmail = form.watch("user_email");
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<SearchField>("user_email");
  const [searchQuery, setSearchQuery] = useState("");
  const [usersSharingEmail, setUsersSharingEmail] = useState<User[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const latestSearchRef = useRef(0);
  const needsUserIdPick = usersSharingEmail.length > 0 && !selectedUserId;
  const hasPickedUser = Boolean(selectedUserId || selectedUserEmail) && !needsUserIdPick;

  const fetchUsers = async (searchText: string, fieldName: SearchField): Promise<void> => {
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
      setUserOptions(groupUsersByField(data, fieldName));
    } catch (error) {
      console.error("Error fetching users:", error);
    } finally {
      if (isLatestSearch()) setLoading(false);
    }
  };

  const handleSearch = (value: string, fieldName: SearchField): void => {
    setSelectedField(fieldName);
    setSearchQuery(value);
    void fetchUsers(value, fieldName);
  };

  const handleSelect = (option: UserOption | null, fieldName: SearchField): void => {
    const [firstUser, ...otherUsers] = option?.users ?? [];
    if (firstUser === undefined) return;
    const isAmbiguous = otherUsers.length > 0;
    form.setValue("user_email", firstUser.user_email);
    form.setValue("user_id", isAmbiguous ? null : firstUser.user_id);
    if (fieldName === "user_email") {
      setUsersSharingEmail(isAmbiguous ? option?.users ?? [] : []);
      return;
    }
    setUsersSharingEmail((users) => (users.some((user) => user.user_id === firstUser.user_id) ? users : []));
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
    setSearchQuery("");
    setUsersSharingEmail([]);
    onCancel();
  };

  const swallowEnter = (event: React.KeyboardEvent): void => {
    if (event.key === "Enter") event.preventDefault();
  };

  const renderUserSearch = (
    fieldName: SearchField,
    placeholder: string,
    controlProps: {
      id: string;
      value: string | null | undefined;
      onChange: (value: string | null | undefined) => void;
    },
    testId?: string,
  ) => {
    const sharedEmailOptions = fieldName === "user_id" ? groupUsersByField(usersSharingEmail, "user_id") : [];
    const items = selectedField === fieldName && searchQuery !== "" ? userOptions : sharedEmailOptions;
    const handleValueChange = (value: string | null) => {
      if (value === null) {
        form.setValue("user_email", null);
        form.setValue("user_id", null);
        setUsersSharingEmail([]);
        return;
      }
      controlProps.onChange(value);
      handleSelect(items.find((option) => option.value === value) ?? null, fieldName);
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
          emptyText="No results"
          loadingText="Loading..."
          inputId={controlProps.id}
        />
      </div>
    );
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleClose()} disablePointerDismissal={isSubmitting}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleSubmit)} noValidate>
            <Alert variant="info" className="mb-4" data-testid="member-existing-users-notice">
              <Info />
              <AlertTitle>
                Search selects from users that already exist. To add someone new, ask a proxy admin to create their
                account first.
              </AlertTitle>
            </Alert>

            <FieldGroup>
              <FormField control={form.control} name="user_email" label="Email">
                {({ id, value, onChange }) =>
                  renderUserSearch("user_email", "Search by email", { id, value, onChange }, "member-email-search")
                }
              </FormField>

              <div className="text-center">OR</div>

              <FormField
                control={form.control}
                name="user_id"
                label="User ID"
                description={needsUserIdPick ? "Multiple users share this email. Pick the user ID to add." : undefined}
              >
                {({ id, value, onChange }) => renderUserSearch("user_id", "Search by user ID", { id, value, onChange })}
              </FormField>

              <FormField control={form.control} name="role" label="Member Role">
                {({ id, value, onChange }) => (
                  <Select items={roles} value={value} onValueChange={(next) => onChange(next as string)}>
                    <SelectTrigger id={id}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {roles.map((role) => (
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
              <Button type="submit" disabled={isSubmitting || !hasPickedUser}>
                {isSubmitting ? <UiLoadingSpinner className="size-4" /> : <UserPlus />}
                {isSubmitting ? "Adding..." : "Add Member"}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default UserSearchModal;
