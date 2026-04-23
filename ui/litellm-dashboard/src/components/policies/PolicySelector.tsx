import React, { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Policy } from "./types";
import { getPoliciesList } from "../networking";

/** Prefix for policy version IDs in request body; must match backend POLICY_VERSION_ID_PREFIX. */
export const POLICY_VERSION_ID_PREFIX = "policy_";

export function policyVersionRef(policyId: string): string {
  return `${POLICY_VERSION_ID_PREFIX}${policyId}`;
}

export function getPolicyOptionEntries(
  policies: Policy[],
): { value: string; label: string }[] {
  return policies
    .filter((policy) => (policy.version_status ?? "draft") !== "draft")
    .map((policy) => {
      const versionNum = policy.version_number ?? 1;
      const status = policy.version_status ?? "draft";
      const label = `${policy.policy_name} — v${versionNum} (${status})${
        policy.description ? ` — ${policy.description}` : ""
      }`;
      const isProduction = status === "production";
      return {
        label,
        value: isProduction
          ? policy.policy_name
          : policy.policy_id
            ? policyVersionRef(policy.policy_id)
            : policy.policy_name,
      };
    });
}

interface PolicySelectorProps {
  onChange: (selectedPolicies: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  disabled?: boolean;
  onPoliciesLoaded?: (policies: Policy[]) => void;
}

const PolicySelector: React.FC<PolicySelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  disabled,
  onPoliciesLoaded,
}) => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    const fetchPolicies = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPoliciesList(accessToken);
        if (response.policies) {
          setPolicies(response.policies);
          onPoliciesLoaded?.(response.policies);
        }
      } catch (error) {
        console.error("Error fetching policies:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, [accessToken, onPoliciesLoaded]);

  const options = useMemo(
    () => getPolicyOptionEntries(policies),
    [policies],
  );

  const selected = value ?? [];
  const filteredOptions = useMemo(
    () =>
      options
        .filter((o) => !selected.includes(o.value))
        .filter((o) =>
          query ? o.label.toLowerCase().includes(query.toLowerCase()) : true,
        ),
    [options, selected, query],
  );

  const labelFor = (v: string) =>
    options.find((o) => o.value === v)?.label ?? v;

  const placeholder = disabled
    ? "Setting policies is a premium feature."
    : loading
      ? "Loading policies…"
      : "Select policies (production or published versions)";

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled || loading}
          className={cn(
            "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",
            className,
          )}
        >
          {selected.length === 0 ? (
            <span className="text-muted-foreground px-1">{placeholder}</span>
          ) : (
            selected.map((v) => (
              <Badge
                key={v}
                variant="secondary"
                className="gap-1 inline-flex items-center"
              >
                {labelFor(v)}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange(selected.filter((s) => s !== v));
                  }}
                  className="inline-flex items-center"
                  aria-label={`Remove ${labelFor(v)}`}
                >
                  <X size={12} />
                </span>
              </Badge>
            ))
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-2"
      >
        <Input
          autoFocus
          placeholder="Search policies…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-8 mb-2"
        />
        <div className="max-h-60 overflow-y-auto">
          {filteredOptions.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No matches
            </div>
          ) : (
            filteredOptions.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
                onClick={() => onChange([...selected, opt.value])}
              >
                {opt.label}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default PolicySelector;
