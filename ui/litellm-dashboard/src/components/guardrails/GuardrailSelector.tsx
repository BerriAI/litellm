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
import { Guardrail } from "./types";
import { getGuardrailsList } from "../networking";

interface GuardrailSelectorProps {
  onChange: (selectedGuardrails: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  disabled?: boolean;
}

/**
 * Multi-select for guardrails. Mirrors the VectorStoreSelector pattern
 * (chip-style trigger + popover suggestion list with search) since shadcn
 * lacks a native multi-select primitive.
 */
const GuardrailSelector: React.FC<GuardrailSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  disabled,
}) => {
  const [guardrails, setGuardrails] = useState<Guardrail[]>([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const selected = useMemo(() => value ?? [], [value]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        if (response.guardrails) setGuardrails(response.guardrails);
      } catch (error) {
        console.error("Error fetching guardrails:", error);
      }
    };
    fetchGuardrails();
  }, [accessToken]);

  const remaining = useMemo(
    () =>
      guardrails
        .filter((g) => g.guardrail_name && !selected.includes(g.guardrail_name))
        .filter((g) =>
          query
            ? (g.guardrail_name ?? "")
                .toLowerCase()
                .includes(query.toLowerCase())
            : true,
        ),
    [guardrails, selected, query],
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",
            className,
          )}
        >
          {selected.length === 0 ? (
            <span className="text-muted-foreground px-1">
              {disabled
                ? "Setting guardrails is a premium feature."
                : "Select guardrails"}
            </span>
          ) : (
            selected.map((name) => (
              <Badge key={name} variant="secondary" className="gap-1">
                {name}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange(selected.filter((s) => s !== name));
                  }}
                  className="inline-flex items-center"
                  aria-label={`Remove ${name}`}
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
          placeholder="Search guardrails…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-8 mb-2"
        />
        <div className="max-h-60 overflow-y-auto">
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No matches
            </div>
          ) : (
            remaining.map((g) => {
              const name = g.guardrail_name as string;
              return (
                <button
                  key={name}
                  type="button"
                  className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
                  onClick={() => onChange([...selected, name])}
                >
                  {name}
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default GuardrailSelector;
