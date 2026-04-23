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
import { getPassThroughEndpointsCall } from "../networking";

interface PassThroughRoutesSelectorProps {
  onChange: (selectedRoutes: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
  teamId?: string | null;
}

interface PassThroughEndpoint {
  path: string;
  methods?: string[];
}

const PassThroughRoutesSelector: React.FC<PassThroughRoutesSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select pass through routes",
  disabled = false,
  teamId,
}) => {
  const [passThroughRoutes, setPassThroughRoutes] = useState<
    Array<{ label: string; value: string }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    const fetchPassThroughRoutes = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPassThroughEndpointsCall(accessToken, teamId);
        if (response.endpoints) {
          const routes = response.endpoints.flatMap(
            (endpoint: PassThroughEndpoint) => {
              const path = endpoint.path;
              const methods = endpoint.methods;

              if (methods && methods.length > 0) {
                return methods.map((method) => ({
                  label: `${method} ${path}`,
                  value: path,
                }));
              }

              return [{ label: path, value: path }];
            },
          );
          setPassThroughRoutes(routes);
        }
      } catch (error) {
        console.error("Error fetching pass through routes:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPassThroughRoutes();
  }, [accessToken, teamId]);

  const selected = value ?? [];
  const filteredOptions = useMemo(
    () =>
      passThroughRoutes
        .filter((o) => !selected.includes(o.value))
        .filter((o) =>
          query ? o.label.toLowerCase().includes(query.toLowerCase()) : true,
        ),
    [passThroughRoutes, selected, query],
  );

  const labelFor = (v: string) =>
    passThroughRoutes.find((o) => o.value === v)?.label ?? v;

  const addFreeform = () => {
    const v = query.trim();
    if (!v) return;
    if (!selected.includes(v)) onChange([...selected, v]);
    setQuery("");
  };

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
            <span className="text-muted-foreground px-1">
              {loading ? "Loading routes…" : placeholder}
            </span>
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
          placeholder="Search or type a custom route…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addFreeform();
            }
          }}
          className="h-8 mb-2"
        />
        <div className="max-h-60 overflow-y-auto">
          {filteredOptions.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {query ? (
                <span>
                  Press Enter to add <span className="font-mono">{query}</span>
                </span>
              ) : (
                "No matches"
              )}
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

export default PassThroughRoutesSelector;
