import React, { useEffect, useState } from "react";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import { fetchOpenAPIRegistry } from "@/components/networking";

export interface OpenAPIKeyTool {
  name: string;
  description: string;
}

export interface OpenAPIRegistryEntry {
  name: string;
  title: string;
  description: string;
  icon_url: string;
  spec_url: string;
  oauth?: {
    authorization_url: string;
    token_url: string;
    pkce: boolean;
    docs_url: string;
  };
  key_tools?: OpenAPIKeyTool[];
}

interface OpenAPIQuickPickerProps {
  accessToken: string | null;
  selectedName: string | null;
  onSelect: (entry: OpenAPIRegistryEntry) => void;
}

const OpenAPIQuickPicker: React.FC<OpenAPIQuickPickerProps> = ({ accessToken, selectedName, onSelect }) => {
  const [apis, setApis] = useState<OpenAPIRegistryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [imgErrors, setImgErrors] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!accessToken) return;
    setLoading(true);
    fetchOpenAPIRegistry(accessToken)
      .then((data) => setApis(data.apis ?? []))
      .catch(() => setApis([]))
      .finally(() => setLoading(false));
  }, [accessToken]);

  const handleImgError = (name: string) => {
    setImgErrors((prev) => new Set(prev).add(name));
  };

  if (loading) {
    return (
      <div className="mb-4">
        <span className="text-sm font-medium">Popular APIs</span>
        <div className="flex justify-center py-6">
          <UiLoadingSpinner className="size-5 text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (apis.length === 0) return null;

  return (
    <div className="mb-4">
      <span className="mb-2 block text-sm font-medium">Popular APIs</span>

      <div className="grid grid-cols-5 gap-2">
        {apis.map((api) => {
          const isSelected = selectedName === api.name;
          const imgFailed = imgErrors.has(api.name);
          return (
            <button
              key={api.name}
              type="button"
              title={api.description}
              onClick={() => onSelect(api)}
              className={cn(
                "flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border p-3 transition-all",
                isSelected ? "border-primary bg-accent shadow-xs" : "border-border hover:bg-accent",
              )}
            >
              {imgFailed ? (
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-muted text-sm font-bold text-muted-foreground">
                  {api.title.charAt(0)}
                </span>
              ) : (
                <img
                  src={api.icon_url}
                  alt={api.title}
                  className="h-7 w-7 object-contain"
                  onError={() => handleImgError(api.name)}
                />
              )}
              <span className="text-center text-xs leading-tight font-medium text-muted-foreground">{api.title}</span>
            </button>
          );
        })}
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        Select an API to pre-fill the spec URL and OAuth 2.0 settings, or enter your own spec URL below.
      </p>
    </div>
  );
};

export default OpenAPIQuickPicker;
