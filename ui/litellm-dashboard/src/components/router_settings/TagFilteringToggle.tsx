import React, { useId } from "react";
import { CONFIG_OWNED_HINT, ConfigOwnedBadge } from "./ConfigOwned";
import { Switch } from "@/components/ui/switch";

interface TagFilteringToggleProps {
  enabled: boolean;
  routerFieldsMetadata: { [key: string]: any };
  onToggle: (enabled: boolean) => void;
  ownedByConfig?: boolean;
}

const TagFilteringToggle: React.FC<TagFilteringToggleProps> = ({
  enabled,
  routerFieldsMetadata,
  onToggle,
  ownedByConfig = false,
}) => {
  const toggleId = useId();

  return (
    <div className="space-y-3 max-w-3xl">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <label htmlFor={toggleId} className="text-xs font-medium text-foreground uppercase tracking-wide">
            {routerFieldsMetadata["enable_tag_filtering"]?.ui_field_name || "Enable Tag Filtering"}
          </label>
          {ownedByConfig && <ConfigOwnedBadge />}
          <p className="text-xs text-muted-foreground mt-0.5">
            {routerFieldsMetadata["enable_tag_filtering"]?.field_description || ""}
            {routerFieldsMetadata["enable_tag_filtering"]?.link && (
              <>
                {" "}
                <a
                  href={routerFieldsMetadata["enable_tag_filtering"].link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-info hover:text-info/80 underline"
                >
                  Learn more
                </a>
              </>
            )}
          </p>
        </div>
        <Switch
          id={toggleId}
          checked={enabled}
          onCheckedChange={onToggle}
          disabled={ownedByConfig}
          title={ownedByConfig ? CONFIG_OWNED_HINT : undefined}
          className="ml-4"
        />
      </div>
    </div>
  );
};

export default TagFilteringToggle;
