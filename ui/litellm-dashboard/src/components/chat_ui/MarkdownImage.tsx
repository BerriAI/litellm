import { ImageIcon } from "lucide-react";
import React, { useState } from "react";

type MarkdownImageProps = Pick<React.ComponentPropsWithoutRef<"img">, "src" | "alt" | "title">;

function describeImageSource(src: string): string {
  const host = URL.canParse(src) ? new URL(src).host : "";
  return host || src;
}

export function MarkdownImage({ src, alt, title }: MarkdownImageProps) {
  const [loaded, setLoaded] = useState(false);

  if (!src) {
    return <span>{alt}</span>;
  }

  if (loaded) {
    // eslint-disable-next-line @next/next/no-img-element -- the host is whatever the model wrote, which next/image cannot allowlist
    return <img src={src} alt={alt ?? ""} title={title} className="max-w-full rounded-md border border-border" />;
  }

  return (
    <button
      type="button"
      onClick={(event) => {
        event.preventDefault();
        setLoaded(true);
      }}
      title={src}
      className="inline-flex max-w-full items-center gap-2 rounded-md border border-border bg-muted px-3 py-1.5 text-sm text-foreground hover:bg-accent"
    >
      <ImageIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="truncate">{alt || "Image"}</span>{" "}
      <span className="truncate text-muted-foreground">{describeImageSource(src)}</span>{" "}
      <span className="shrink-0 font-medium">Load image</span>
    </button>
  );
}
