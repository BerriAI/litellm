import { useRef, useState } from "react";
import { Info, UserPlus } from "lucide-react";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { useForm } from "react-hook-form";
import { userFilterUICall } from "@/components/networking";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
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
  user_email: string | undefined;
  user_id: string | undefined;
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

  const debouncedSearch = useDebouncedCallback(
    (text: string, fieldName: "user_email" | "user_id") => fetchUsers(text, fieldName),
    { wait: DEBOUNCE_WAIT_MS },
  );

  const handleSearch = (value: string, fieldName: "user_email" | "user_id"): void => {
    setSelectedField(fieldName);
    debouncedSearch(value, fieldName);
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

  const optionsFor = (fieldName: "user_email" | "user_id", value: string | undefined): UserOption[] => {
    const visible = selectedField === fieldName ? userOptions : [];
    if (value == null || value === "" || visible.some((option) => option.value === value)) return visible;
    return [{ label: value, value, user: null }, ...visible];
  };

  const renderUserSearch = (
    fieldName: "user_email" | "user_id",
    placeholder: string,
    controlProps: { id: string; value: string | undefined; onChange: (value: string | undefined) => void },
    testId?: string,
  ) => {
    const items = optionsFor(fieldName, controlProps.value);
    const selected = items.find((option) => option.value === controlProps.value) ?? null;
    return (
      <div data-testid={testId}>
        <Combobox
          items={items}
          value={selected}
          // @ts-expect-error TS2322 -- Combobox.Root narrows autoHighlight to boolean; the AriaCombobox it wraps
          // accepts "always", the only value that highlights a list this component filters server-side
          autoHighlight="always"
          filter={null}
          onValueChange={(option: UserOption | null) => {
            controlProps.onChange(option?.value);
            handleSelect(option);
          }}
          onInputValueChange={(text: string) => handleSearch(text, fieldName)}
          isItemEqualToValue={(a: UserOption, b: UserOption) => a.value === b.value}
          itemToStringLabel={(option: UserOption) => option.label}
        >
          <ComboboxInput
            id={controlProps.id}
            placeholder={placeholder}
            showClear={selected !== null}
            onKeyDown={swallowEnter}
          />
          <ComboboxContent>
            <ComboboxEmpty>{loading ? "Loading..." : "No results"}</ComboboxEmpty>
            <ComboboxList>
              {(option: UserOption) => (
                <ComboboxItem key={option.value} value={option}>
                  {option.label}
                </ComboboxItem>
              )}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
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

              <FormField control={form.control} name="user_id" label="User ID">
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
              <Button type="submit" disabled={isSubmitting}>
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
