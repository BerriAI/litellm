import { Badge } from "@/components/ui/badge";

export type ModelCustomMetadataValue = string | number | boolean;

interface ModelCustomMetadataProps {
  customTags?: string[] | null;
  customMetadata?: Record<string, ModelCustomMetadataValue> | null;
}

export function ModelCustomMetadata({ customTags, customMetadata }: ModelCustomMetadataProps) {
  const tags = customTags ?? [];
  const entries = Object.entries(customMetadata ?? {});
  return (
    <>
      {tags.length > 0 && (
        <div className="col-span-2">
          <p className="font-medium">Tags:</p>
          <div className="flex flex-wrap gap-1 mt-1">
            {tags.map((tag) => (
              <Badge key={tag} variant="outline">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      )}
      {entries.length > 0 && (
        <div className="col-span-2">
          <p className="font-medium">Custom Metadata:</p>
          <dl className="mt-1 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1">
            {entries.map(([key, value]) => (
              <div key={key} className="contents">
                <dt className="text-muted-foreground">{key}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </>
  );
}
