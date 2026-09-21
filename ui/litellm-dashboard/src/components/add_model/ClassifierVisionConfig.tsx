import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import React from "react";

import type { ClassifierLLMConfigWire } from "./build_complexity_router_config";

export const DEFAULT_CLASSIFIER_VISION_ENABLED = false;
export const DEFAULT_CLASSIFIER_VISION_MAX_IMAGES = 1;

const MAX_IMAGES_ID = "classifier-vision-max-images";

interface ClassifierVisionConfigProps {
  value: ClassifierLLMConfigWire;
  onChange: (value: ClassifierLLMConfigWire) => void;
}

const ClassifierVisionConfig: React.FC<ClassifierVisionConfigProps> = ({ value, onChange }) => {
  const [draftMaxImages, setDraftMaxImages] = React.useState<string | null>(null);
  const enabled = value.vision?.enabled ?? DEFAULT_CLASSIFIER_VISION_ENABLED;

  const handleMaxImagesChange = (raw: string): void => {
    setDraftMaxImages(raw);
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    onChange({
      ...value,
      vision: { ...value.vision, enabled, max_images: Math.max(1, Math.round(parsed)) },
    });
  };

  return (
    <div className="space-y-2 rounded-md border border-border p-3">
      <div className="flex items-center gap-2">
        <Switch
          checked={enabled}
          onCheckedChange={(visionEnabled): void => {
            if (!visionEnabled) {
              const { vision: _vision, ...withoutVision } = value;
              onChange(withoutVision);
              return;
            }
            onChange({
              ...value,
              vision: {
                ...value.vision,
                enabled: true,
                max_images: value.vision?.max_images ?? DEFAULT_CLASSIFIER_VISION_MAX_IMAGES,
              },
            });
          }}
          aria-label="Use images for classification"
        />
        <strong className="font-semibold">Use images for classification</strong>
      </div>
      <span className="block text-xs text-muted-foreground">
        Send inline image data to the classifier so it can choose a tier from what the image shows.
      </span>
      {enabled && (
        <div>
          <Label htmlFor={MAX_IMAGES_ID} className="block mb-1 font-semibold">
            Maximum images per request
          </Label>
          <Input
            id={MAX_IMAGES_ID}
            type="text"
            inputMode="numeric"
            value={draftMaxImages ?? String(value.vision?.max_images ?? DEFAULT_CLASSIFIER_VISION_MAX_IMAGES)}
            onChange={(event) => handleMaxImagesChange(event.target.value)}
            onBlur={() => setDraftMaxImages(null)}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
};

export default ClassifierVisionConfig;
