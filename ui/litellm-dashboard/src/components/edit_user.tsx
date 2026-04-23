import { useEffect } from "react";
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
import NumericalInput from "./shared/numerical_input";
import BudgetDurationDropdown from "./common_components/budget_duration_dropdown";
import { Controller, FormProvider, useForm } from "react-hook-form";

interface EditUserModalProps {
  visible: boolean;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  user: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSubmit: (data: any) => void;
}

interface EditUserValues {
  user_email: string;
  user_id: string;
  user_role: string;
  spend: number | null;
  max_budget: number | null;
  budget_duration: string | null;
}

function LabeledRow({
  id,
  label,
  help,
  tooltip,
  children,
}: {
  id?: string;
  label: string;
  help?: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-3 gap-3 items-start mb-4">
      <Label htmlFor={id} className="pt-2" title={tooltip}>
        {label}
      </Label>
      <div className="col-span-2 space-y-1">
        {children}
        {help && <p className="text-xs text-muted-foreground">{help}</p>}
      </div>
    </div>
  );
}

const EditUserModal: React.FC<EditUserModalProps> = ({
  visible,
  possibleUIRoles,
  onCancel,
  user,
  onSubmit,
}) => {
  const form = useForm<EditUserValues>({
    defaultValues: {
      user_email: "",
      user_id: "",
      user_role: "",
      spend: null,
      max_budget: null,
      budget_duration: null,
    },
    mode: "onSubmit",
  });

  useEffect(() => {
    if (user) {
      form.reset({
        user_email: user.user_email ?? "",
        user_id: user.user_id ?? "",
        user_role: user.user_role ?? "",
        spend: user.spend ?? null,
        max_budget: user.max_budget ?? null,
        budget_duration: user.budget_duration ?? null,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const handleCancel = () => {
    form.reset();
    onCancel();
  };

  const handleEditSubmit = form.handleSubmit((values) => {
    onSubmit(values);
    form.reset();
    onCancel();
  });

  if (!user) return null;

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit User {user.user_id}</DialogTitle>
          <DialogDescription className="sr-only">
            Edit user attributes including role, budget, and rate limits.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleEditSubmit}>
            <LabeledRow id="user_email" label="User Email" tooltip="Email of the User">
              <Input id="user_email" {...form.register("user_email")} />
            </LabeledRow>

            {/* hidden user_id field */}
            <input type="hidden" {...form.register("user_id")} />

            <LabeledRow label="User Role">
              <Controller
                control={form.control}
                name="user_role"
                render={({ field }) => (
                  <Select
                    value={field.value ?? ""}
                    onValueChange={(v) => field.onChange(v)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {possibleUIRoles &&
                        Object.entries(possibleUIRoles).map(
                          ([role, { ui_label, description }]) => (
                            <SelectItem key={role} value={role} title={ui_label}>
                              <span className="flex">
                                {ui_label}{" "}
                                <span className="ml-2 text-muted-foreground text-xs">
                                  {description}
                                </span>
                              </span>
                            </SelectItem>
                          ),
                        )}
                    </SelectContent>
                  </Select>
                )}
              />
            </LabeledRow>

            <LabeledRow
              id="spend"
              label="Spend (USD)"
              tooltip="(float) - Spend of all LLM calls completed by this user"
              help="Across all keys (including keys with team_id)."
            >
              <Input
                id="spend"
                type="number"
                step={0.01}
                min={0}
                {...form.register("spend", { valueAsNumber: true })}
              />
            </LabeledRow>

            <LabeledRow
              id="max_budget"
              label="User Budget (USD)"
              tooltip="(float) - Maximum budget of this user"
              help="Maximum budget of this user."
            >
              <Controller
                control={form.control}
                name="max_budget"
                render={({ field }) => (
                  <NumericalInput
                    min={0}
                    step={0.01}
                    value={field.value ?? ""}
                    onChange={(v: number | null | undefined) =>
                      field.onChange(v ?? null)
                    }
                  />
                )}
              />
            </LabeledRow>

            <LabeledRow label="Reset Budget">
              <Controller
                control={form.control}
                name="budget_duration"
                render={({ field }) => (
                  <BudgetDurationDropdown
                    value={field.value ?? undefined}
                    onChange={(value) =>
                      field.onChange((value as string) ?? null)
                    }
                  />
                )}
              />
            </LabeledRow>

            <DialogFooter>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default EditUserModal;
