import React from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export interface CustomerRateLimitEntry {
  id: string;
  customer: string;
  rpm_limit: number | null;
  tpm_limit: number | null;
}

let nextRowId = 0;
const newRowId = (): string => `customer-row-${nextRowId++}`;

export interface CustomerRateLimits {
  customer_rpm_limit: Record<string, number>;
  customer_tpm_limit: Record<string, number>;
}

export const customerRowsToLimits = (rows: CustomerRateLimitEntry[]): CustomerRateLimits => {
  const customer_rpm_limit: Record<string, number> = {};
  const customer_tpm_limit: Record<string, number> = {};
  rows.forEach(({ customer, rpm_limit, tpm_limit }) => {
    const name = customer.trim();
    if (!name) return;
    if (typeof rpm_limit === "number") customer_rpm_limit[name] = rpm_limit;
    if (typeof tpm_limit === "number") customer_tpm_limit[name] = tpm_limit;
  });
  return { customer_rpm_limit, customer_tpm_limit };
};

const toNumberMap = (raw: unknown): Record<string, number> => {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, number> = {};
  Object.entries(raw as Record<string, unknown>).forEach(([customer, limit]) => {
    if (typeof limit === "number") out[customer] = limit;
  });
  return out;
};

export const customerLimitsToRows = (
  customerRpmLimit?: unknown,
  customerTpmLimit?: unknown,
): CustomerRateLimitEntry[] => {
  const rpm = toNumberMap(customerRpmLimit);
  const tpm = toNumberMap(customerTpmLimit);
  return Array.from(new Set([...Object.keys(rpm), ...Object.keys(tpm)])).map((customer) => ({
    id: newRowId(),
    customer,
    rpm_limit: rpm[customer] ?? null,
    tpm_limit: tpm[customer] ?? null,
  }));
};

interface CustomerRateLimitEditorProps {
  value: CustomerRateLimitEntry[];
  onChange: (v: CustomerRateLimitEntry[]) => void;
}

const parseLimit = (raw: string): number | null => (raw === "" ? null : Number(raw));

export function CustomerRateLimitEditor({ value, onChange }: CustomerRateLimitEditorProps) {
  const { t } = useTranslation();
  const addRow = () => {
    onChange([...value, { id: newRowId(), customer: "", rpm_limit: null, tpm_limit: null }]);
  };

  const removeRow = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const updateRow = (idx: number, field: keyof CustomerRateLimitEntry, fieldValue: string | number | null) => {
    onChange(value.map((row, i) => (i === idx ? { ...row, [field]: fieldValue } : row)));
  };

  return (
    <div>
      {value.map((row, idx) => (
        <div key={row.id} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
          <Input
            aria-label={t("keyTeamHelpers.customerRateLimitEditor.customer")}
            value={row.customer}
            onChange={(e) => updateRow(idx, "customer", e.target.value)}
            placeholder={t("keyTeamHelpers.customerRateLimitEditor.customerPlaceholder")}
            style={{ width: 180 }}
          />
          <Input
            aria-label={t("keyTeamHelpers.customerRateLimitEditor.rpmLimit")}
            type="number"
            min={0}
            value={row.rpm_limit ?? ""}
            onChange={(e) => updateRow(idx, "rpm_limit", parseLimit(e.target.value))}
            placeholder={t("keyTeamHelpers.customerRateLimitEditor.rpmPlaceholder")}
            style={{ width: 120 }}
          />
          <Input
            aria-label={t("keyTeamHelpers.customerRateLimitEditor.tpmLimit")}
            type="number"
            min={0}
            value={row.tpm_limit ?? ""}
            onChange={(e) => updateRow(idx, "tpm_limit", parseLimit(e.target.value))}
            placeholder={t("keyTeamHelpers.customerRateLimitEditor.tpmPlaceholder")}
            style={{ width: 120 }}
          />
          <Button
            variant="destructive"
            size="sm"
            aria-label={t("keyTeamHelpers.customerRateLimitEditor.removeCustomerLimit")}
            onClick={() => removeRow(idx)}
          >
            ✕
          </Button>
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        onClick={(e) => {
          e.preventDefault();
          addRow();
        }}
      >
        {t("keyTeamHelpers.customerRateLimitEditor.addCustomerLimit")}
      </Button>
    </div>
  );
}
