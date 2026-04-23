import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchOpenAPIRegistry } from "../networking";

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

const OpenAPIQuickPicker: React.FC<OpenAPIQuickPickerProps> = ({
  accessToken,
  selectedName,
  onSelect,
}) => {
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
        <span className="text-sm font-medium text-foreground">
          Popular APIs
        </span>
        <div className="flex justify-center py-6">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (apis.length === 0) return null;

  return (
    <div className="mb-4">
      <span className="text-sm font-medium text-foreground block mb-2">
        Popular APIs
      </span>

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
                "flex flex-col items-center gap-1.5 p-3 rounded-lg border transition-all cursor-pointer",
                isSelected
                  ? "border-primary bg-primary/10 shadow-sm"
                  : "border-border hover:border-primary/50 hover:bg-muted",
              )}
            >
              {imgFailed ? (
                <span className="w-7 h-7 rounded-full bg-muted flex items-center justify-center text-sm font-bold text-foreground">
                  {api.title.charAt(0)}
                </span>
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={api.icon_url}
                  alt={api.title}
                  className="w-7 h-7 object-contain"
                  onError={() => handleImgError(api.name)}
                />
              )}
              <span className="text-xs text-muted-foreground text-center leading-tight font-medium">
                {api.title}
              </span>
            </button>
          );
        })}
      </div>

      <p className="text-xs text-muted-foreground mt-2">
        Select an API to pre-fill the spec URL and OAuth 2.0 settings, or enter
        your own spec URL below.
      </p>
    </div>
  );
};

export default OpenAPIQuickPicker;
