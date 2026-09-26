import { ImageIcon } from "lucide-react";
import React, { useState } from "react";

import { Button } from "@/components/ui/button";

type MarkdownImageProps = Pick<React.ComponentPropsWithoutRef<"img">, "src" | "alt" | "title">;

type ImageSource = { readonly label: string; readonly tooltip: string | undefined };

function describeImageSource(src: string): ImageSource {
  if (src.startsWith("data:")) {
    return { label: "inline image", tooltip: undefined };
  }
  const host = URL.canParse(src) ? new URL(src).host : "";
  return { label: host || src, tooltip: src };
}

export function MarkdownImage({ src, alt, title }: MarkdownImageProps) {
  const [loaded, setLoaded] = useState(false);

  if (typeof src !== "string" || src === "") {
    return <span>{alt}</span>;
  }

  if (loaded) {
    // eslint-disable-next-line @next/next/no-img-element -- the host is whatever the model wrote, which next/image cannot allowlist
    return <img src={src} alt={alt ?? ""} title={title} className="max-w-full rounded-md border border-border" />;
  }

  const source = describeImageSource(src);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={(event) => {
        event.preventDefault();
        setLoaded(true);
      }}
      title={source.tooltip}
      className="max-w-full font-normal"
    >
      <ImageIcon className="text-muted-foreground" aria-hidden="true" />
      <span className="truncate">{alt || "Image"}</span>{" "}
      <span className="truncate text-muted-foreground">{source.label}</span>{" "}
      <span className="shrink-0 font-medium">Load image</span>
    </Button>
  );
}
