"use client";

import { ArrowLeft, MoreVertical, Pencil, Trash2 } from "lucide-react";
import type { ReactNode } from "react";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import ObjectPermissionsView from "@/components/object_permissions_view";
import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import CopyButton from "@/components/shared/CopyButton";
import { PageHeader, PageHeaderTitle } from "@/components/shared/PageHeader";
import { ModelsCell, MoneyCell } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "@/lib/toast";

import { CustomerForm } from "./CustomerForm";
import { CustomerBlockAction, CustomerBlockedBadge } from "./CustomersTableColumns";
import type { CustomerFormValues } from "./customerPayload";
import { useSaveCustomer } from "./useSaveCustomer";

type CustomerTab = "overview" | "settings";

type CustomerInfoViewProps = {
  customer: EndUser;
  accessToken: string | null;
  canModify: boolean;
  premiumUser: boolean;
  isEdit: boolean;
  tab: CustomerTab;
  isBlocking: boolean;
  hasSharedBudget: boolean | undefined;
  onTabChange: (tab: CustomerTab) => void;
  onBack: () => void;
  onEdit: () => void;
  onCancelEdit: () => void;
  onDelete: (customer: EndUser) => void;
  onSetBlocked: (customer: EndUser) => void;
};

function OverviewField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="break-words text-sm">{children}</dd>
    </div>
  );
}

function CustomerModelFields({ customer }: { customer: EndUser }) {
  return (
    <>
      <OverviewField label="Models">
        <ModelsCell models={customer.models} unrestrictedLabel="All models the key can use" />
      </OverviewField>
      <OverviewField label="Allowed Model Region">
        {customer.allowed_model_region?.toUpperCase() ?? "Any"}
      </OverviewField>
      <OverviewField label="Default Model">{customer.default_model ?? "None"}</OverviewField>
    </>
  );
}

function CustomerOverview({ customer, accessToken }: Pick<CustomerInfoViewProps, "customer" | "accessToken">) {
  const budget = customer.litellm_budget_table;
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">Budget Status</p>
            <div className="mt-2 space-y-2 text-sm text-foreground">
              <div className="text-xl font-semibold [&>span]:text-left">
                <MoneyCell value={customer.spend} decimals={2} showZero />
              </div>
              <div className="flex gap-1">
                of{" "}
                <span className="[&>span]:text-left">
                  <MoneyCell value={budget?.max_budget} decimals={2} emptyText="Unlimited" showZero />
                </span>
              </div>
              <p className="text-muted-foreground">Reset: {getBudgetDurationLabel(budget?.budget_duration)}</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">Rate Limits</p>
            <dl className="mt-2 space-y-3">
              <OverviewField label="TPM">{budget?.tpm_limit ?? "Unlimited"}</OverviewField>
              <OverviewField label="RPM">{budget?.rpm_limit ?? "Unlimited"}</OverviewField>
            </dl>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">Models</p>
            <dl className="mt-2 space-y-3">
              <CustomerModelFields customer={customer} />
            </dl>
          </CardContent>
        </Card>
      </div>
      <ObjectPermissionsView
        objectPermission={customer.object_permission}
        accessToken={accessToken}
        entityType="customer"
      />
    </div>
  );
}

function CustomerSettings({ customer, accessToken }: Pick<CustomerInfoViewProps, "customer" | "accessToken">) {
  const budget = customer.litellm_budget_table;
  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <OverviewField label="Alias">{customer.alias || "Not set"}</OverviewField>
        <OverviewField label="Budget ID">{customer.budget_id ?? "No budget"}</OverviewField>
        <OverviewField label="Max Budget">
          <span className="[&>span]:text-left">
            <MoneyCell value={budget?.max_budget} decimals={2} emptyText="Unlimited" showZero />
          </span>
        </OverviewField>
        <OverviewField label="Reset Period">{getBudgetDurationLabel(budget?.budget_duration)}</OverviewField>
        <OverviewField label="TPM">{budget?.tpm_limit ?? "Unlimited"}</OverviewField>
        <OverviewField label="RPM">{budget?.rpm_limit ?? "Unlimited"}</OverviewField>
        <CustomerModelFields customer={customer} />
      </dl>
      <ObjectPermissionsView
        objectPermission={customer.object_permission}
        entityType="customer"
        variant="inline"
        className="border-t pt-4"
        accessToken={accessToken}
      />
    </div>
  );
}

function CustomerActions({
  customer,
  premiumUser,
  isBlocking,
  onDelete,
  onSetBlocked,
}: Pick<CustomerInfoViewProps, "customer" | "premiumUser" | "isBlocking" | "onDelete" | "onSetBlocked">) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="outline" size="icon" aria-label="More customer actions" />}>
        <MoreVertical className="size-3.5" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-auto">
        <CustomerBlockAction
          customer={customer}
          premiumUser={premiumUser}
          isBlocking={isBlocking}
          onSetBlocked={onSetBlocked}
        />
        <DropdownMenuItem variant="destructive" onClick={() => onDelete(customer)}>
          <Trash2 className="size-3.5" />
          Delete Customer
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default function CustomerInfoView({
  customer,
  accessToken,
  canModify,
  hasSharedBudget,
  isEdit,
  tab,
  onTabChange,
  onBack,
  onEdit,
  onCancelEdit,
  ...actions
}: CustomerInfoViewProps) {
  const { saveCustomer, isSaving } = useSaveCustomer(customer);
  const handleSave = async (values: CustomerFormValues) => {
    try {
      await saveCustomer(values);
      toast.success("Customer updated");
      onCancelEdit();
    } catch (error) {
      toast.fromError(error);
    }
  };
  const showEdit = canModify && isEdit;
  const hasDistinctAlias = Boolean(customer.alias && customer.alias !== customer.user_id);

  return (
    <>
      <PageHeader>
        <Button variant="ghost" className="mb-4" onClick={onBack}>
          <ArrowLeft />
          Back to Customers
        </Button>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <PageHeaderTitle className="min-w-0 break-words [&_svg]:size-4">
                {customer.alias || customer.user_id}
                <CopyButton
                  value={customer.alias || customer.user_id}
                  label={hasDistinctAlias ? "Copy Customer Alias" : "Copy Customer ID"}
                  iconClassName="size-4"
                />
              </PageHeaderTitle>
              {customer.blocked && <CustomerBlockedBadge />}
            </div>
            {hasDistinctAlias && (
              <div className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 break-words text-muted-foreground">Customer ID: {customer.user_id}</span>
                <CopyButton value={customer.user_id} label="Copy Customer ID" iconClassName="size-3.5" />
              </div>
            )}
          </div>
          {canModify && !showEdit && <CustomerActions {...actions} customer={customer} />}
        </div>
      </PageHeader>
      <Tabs
        value={showEdit ? "settings" : tab}
        onValueChange={(value) => {
          if (value === "overview" || value === "settings") onTabChange(value);
        }}
      >
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
            Settings
          </TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="pt-4">
          <CustomerOverview customer={customer} accessToken={accessToken} />
        </TabsContent>
        <TabsContent value="settings" className="pt-4">
          <Card className="max-h-[65vh] overflow-y-auto">
            <CardContent>
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-medium">Customer Settings</h3>
                {canModify && !showEdit && (
                  <Button variant="outline" onClick={onEdit}>
                    <Pencil />
                    Edit Settings
                  </Button>
                )}
              </div>
              {showEdit ? (
                <CustomerForm
                  customer={customer}
                  isEdit
                  isSaving={isSaving}
                  hasSharedBudget={hasSharedBudget}
                  onSubmit={handleSave}
                  onCancel={onCancelEdit}
                />
              ) : (
                <CustomerSettings customer={customer} accessToken={accessToken} />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
