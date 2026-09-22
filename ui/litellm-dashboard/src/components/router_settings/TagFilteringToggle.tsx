import React, { useId } from "react";
import { ConfigOwnedField } from "@/components/shared/ConfigOwnedField";
import { Switch } from "@/components/ui/switch";

interface TagFilteringToggleProps {
  enabled: boolean;
  routerFieldsMetadata: { [key: string]: any };
  disabled?: boolean;
  onToggle: (enabled: boolean) => void;
}

const TagFilteringToggle: React.FC<TagFilteringToggleProps> = ({
  enabled,
  routerFieldsMetadata,
  disabled = false,
  onToggle,
}) => {
  const toggleId = useId();

  return (
    <div className="space-y-3 max-w-3xl">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <label htmlFor={toggleId} className="text-xs font-medium text-foreground uppercase tracking-wide">
            {routerFieldsMetadata["enable_tag_filtering"]?.ui_field_name || "Enable Tag Filtering"}
          </label>
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
        <ConfigOwnedField frozen={disabled} className="ml-4 inline-flex">
          <Switch id={toggleId} checked={enabled} disabled={disabled} onCheckedChange={onToggle} />
        </ConfigOwnedField>
      </div>
    </div>
  );
};

export default TagFilteringToggle;
