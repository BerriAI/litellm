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
import { VectorStore } from "./types";
import { vectorStoreListCall } from "../networking";

interface VectorStoreSelectorProps {
  onChange: (selectedVectorStores: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

/**
 * Multi-select for vector stores. Uses the shadcn Popover + chip pattern
 * (shared with GuardrailSelector) since shadcn has no native multi-select.
 */
const VectorStoreSelector: React.FC<VectorStoreSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select vector stores",
  disabled = false,
}) => {
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const selected = useMemo(() => value ?? [], [value]);

  useEffect(() => {
    const fetchVectorStores = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await vectorStoreListCall(accessToken);
        if (response.data) {
          setVectorStores(response.data);
        }
      } catch (error) {
        console.error("Error fetching vector stores:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchVectorStores();
  }, [accessToken]);

  const optionFor = (id: string) => vectorStores.find((s) => s.vector_store_id === id);
  const labelFor = (store: VectorStore) =>
    `${store.vector_store_name || store.vector_store_id} (${store.vector_store_id})`;

  const remaining = useMemo(
    () =>
      vectorStores
        .filter((s) => !selected.includes(s.vector_store_id))
        .filter((s) => {
          if (!query) return true;
          const label = labelFor(s).toLowerCase();
          return label.includes(query.toLowerCase());
        }),
    [vectorStores, selected, query],
  );

  return (
    <div>
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
                {loading ? "Loading…" : placeholder}
              </span>
            ) : (
              selected.map((id) => {
                const store = optionFor(id);
                const label = store
                  ? store.vector_store_name || store.vector_store_id
                  : id;
                return (
                  <Badge
                    key={id}
                    variant="secondary"
                    className="gap-1"
                    title={store?.vector_store_description || id}
                  >
                    {label}
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        onChange(selected.filter((s) => s !== id));
                      }}
                      className="inline-flex items-center"
                      aria-label={`Remove ${label}`}
                    >
                      <X size={12} />
                    </span>
                  </Badge>
                );
              })
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="w-[var(--radix-popover-trigger-width)] p-2"
        >
          <Input
            autoFocus
            placeholder="Search vector stores…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 mb-2"
          />
          <div className="max-h-60 overflow-y-auto">
            {loading ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                Loading…
              </div>
            ) : remaining.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No matches
              </div>
            ) : (
              remaining.map((store) => (
                <button
                  key={store.vector_store_id}
                  type="button"
                  className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
                  onClick={() => {
                    onChange([...selected, store.vector_store_id]);
                    setQuery("");
                  }}
                  title={store.vector_store_description || store.vector_store_id}
                >
                  {labelFor(store)}
                </button>
              ))
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default VectorStoreSelector;
