import { useCallback, useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import debounce from "lodash/debounce";
import { UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { userFilterUICall } from "@/components/networking";

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}

interface Role {
  label: string;
  value: string;
  description: string;
}

interface FormValues {
  user_email: string;
  user_id: string;
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

// Async searchable combobox that lets the user type a query and pick from
// server-fetched results. Internally a Popover wrapping an Input + list.
interface UserComboboxProps {
  placeholder: string;
  value: string;
  loading: boolean;
  options: UserOption[];
  dataTestId?: string;
  onSearch: (text: string) => void;
  onSelect: (value: string, option: UserOption) => void;
  onClear: () => void;
}

function UserCombobox({
  placeholder,
  value,
  loading,
  options,
  dataTestId,
  onSearch,
  onSelect,
  onClear,
}: UserComboboxProps) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          data-testid={dataTestId}
          aria-expanded={open}
          className={cn(
            "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          )}
        >
          <span className={cn(!value && "text-muted-foreground", "truncate")}>
            {value || placeholder}
          </span>
          {value && (
            <span
              role="button"
              tabIndex={0}
              aria-label="Clear"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="text-muted-foreground"
            >
              ×
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width] max-w-none">
        <div className="flex items-center border-b border-border p-2">
          <Input
            value={text}
            placeholder={placeholder}
            onChange={(e) => {
              setText(e.target.value);
              onSearch(e.target.value);
            }}
            className="h-8 border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          />
        </div>
        <div className="max-h-60 overflow-y-auto">
          {loading ? (
            <div className="p-3 text-sm text-muted-foreground text-center">
              Loading...
            </div>
          ) : options.length === 0 ? (
            <div className="p-3 text-sm text-muted-foreground text-center">
              No results
            </div>
          ) : (
            options.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={cn(
                  "block w-full text-left text-sm px-3 py-2 hover:bg-muted",
                  opt.value === value && "bg-muted font-medium",
                )}
                onClick={() => {
                  onSelect(opt.value, opt);
                  setText("");
                  setOpen(false);
                }}
              >
                {opt.label}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
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
      description:
        "Admin role. Can create team keys, add members, and manage settings.",
    },
    {
      label: "user",
      value: "user",
      description: "User role. Can view team info, but not manage it.",
    },
  ],
  defaultRole = "user",
  teamId,
}) => {
  const methods = useForm<FormValues>({
    defaultValues: {
      user_email: "",
      user_id: "",
      role: defaultRole,
    },
  });
  const { control, handleSubmit, reset, setValue } = methods;

  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<"user_email" | "user_id">(
    "user_email",
  );
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchUsers = async (
    searchText: string,
    fieldName: "user_email" | "user_id",
  ): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.append(fieldName, searchText);
      if (teamId) params.append("team_id", teamId);
      if (accessToken == null) return;
      const response = await userFilterUICall(accessToken, params);
      const data: User[] = response;
      const options: UserOption[] = data.map((user) => ({
        label:
          fieldName === "user_email" ? `${user.user_email}` : `${user.user_id}`,
        value:
          fieldName === "user_email" ? user.user_email : user.user_id,
        user,
      }));
      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
    } finally {
      setLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce(
      (text: string, fieldName: "user_email" | "user_id") =>
        fetchUsers(text, fieldName),
      300,
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const handleSearch = (
    value: string,
    fieldName: "user_email" | "user_id",
  ): void => {
    setSelectedField(fieldName);
    debouncedSearch(value, fieldName);
  };

  const handleSelect = (_value: string, option: UserOption): void => {
    const user = option.user;
    setValue("user_email", user.user_email);
    setValue("user_id", user.user_id);
  };

  const submit = handleSubmit(async (values) => {
    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  });

  const handleClose = (): void => {
    reset({ user_email: "", user_id: "", role: defaultRole });
    setUserOptions([]);
    onCancel();
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o && !isSubmitting ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <FormProvider {...methods}>
          <form onSubmit={submit} className="space-y-4">
            <div className="grid grid-cols-[120px_1fr] gap-3 items-center">
              <Label htmlFor="user_email" className="text-left">
                Email
              </Label>
              <Controller
                control={control}
                name="user_email"
                render={({ field }) => (
                  <UserCombobox
                    placeholder="Search by email"
                    value={field.value}
                    loading={loading && selectedField === "user_email"}
                    options={
                      selectedField === "user_email" ? userOptions : []
                    }
                    dataTestId="member-email-search"
                    onSearch={(t) => handleSearch(t, "user_email")}
                    onSelect={handleSelect}
                    onClear={() => {
                      setValue("user_email", "");
                      setValue("user_id", "");
                    }}
                  />
                )}
              />
            </div>

            <div className="text-center">OR</div>

            <div className="grid grid-cols-[120px_1fr] gap-3 items-center">
              <Label htmlFor="user_id" className="text-left">
                User ID
              </Label>
              <Controller
                control={control}
                name="user_id"
                render={({ field }) => (
                  <UserCombobox
                    placeholder="Search by user ID"
                    value={field.value}
                    loading={loading && selectedField === "user_id"}
                    options={
                      selectedField === "user_id" ? userOptions : []
                    }
                    onSearch={(t) => handleSearch(t, "user_id")}
                    onSelect={handleSelect}
                    onClear={() => {
                      setValue("user_email", "");
                      setValue("user_id", "");
                    }}
                  />
                )}
              />
            </div>

            <div className="grid grid-cols-[120px_1fr] gap-3 items-center">
              <Label htmlFor="role" className="text-left">
                Member Role
              </Label>
              <Controller
                control={control}
                name="role"
                render={({ field }) => (
                  <Select
                    value={field.value}
                    onValueChange={field.onChange}
                  >
                    <SelectTrigger id="role" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {roles.map((role) => (
                        <SelectItem key={role.value} value={role.value}>
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span>
                                  <span className="font-medium">
                                    {role.label}
                                  </span>
                                  <span className="ml-2 text-muted-foreground text-sm">
                                    - {role.description}
                                  </span>
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                {role.description}
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>

            <div className="text-right mt-4">
              <Button type="submit" disabled={isSubmitting}>
                <UserPlus className="h-4 w-4" />
                {isSubmitting ? "Adding..." : "Add Member"}
              </Button>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default UserSearchModal;
