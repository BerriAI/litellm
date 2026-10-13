"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { toast } from "@/lib/toast";
import { CustomerForm } from "./CustomerForm";
import type { CustomerFormValues } from "./customerPayload";
import { useSaveCustomer } from "./useSaveCustomer";

type CreateCustomerModalProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: (customerId: string) => void;
};

function CreateCustomerFields({ onOpenChange, onCreated }: Omit<CreateCustomerModalProps, "open">) {
  const { saveCustomer, isSaving } = useSaveCustomer(null);
  const handleSubmit = async (values: CustomerFormValues) => {
    try {
      const customer = await saveCustomer(values);
      toast.success("Customer created successfully");
      onOpenChange(false);
      onCreated?.(customer.user_id);
    } catch (error) {
      toast.fromError(error);
    }
  };

  return <CustomerForm onSubmit={handleSubmit} onCancel={() => onOpenChange(false)} isSaving={isSaving} />;
}

export function CreateCustomerModal({ open, onOpenChange, onCreated }: CreateCustomerModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Create Customer</DialogTitle>
        </DialogHeader>
        <CreateCustomerFields onOpenChange={onOpenChange} onCreated={onCreated} />
      </DialogContent>
    </Dialog>
  );
}
