import React, { useEffect, useState } from "react";
import { Code, Download, FileImage, FileText, Loader2 } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface ContainerFileCitation {
  type: "container_file_citation";
  container_id: string;
  file_id: string;
  filename: string;
  start_index: number;
  end_index: number;
}

interface CodeInterpreterOutputProps {
  code?: string;
  containerId?: string;
  annotations?: ContainerFileCitation[];
  accessToken: string;
}

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif"] as const;

function isImageFilename(filename: string | undefined): boolean {
  if (!filename) {
    return false;
  }
  const lower = filename.toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

const CodeInterpreterOutput: React.FC<CodeInterpreterOutputProps> = ({ code, annotations = [], accessToken }) => {
  const syntaxTheme = useSyntaxTheme(coy);
  const [imageUrls, setImageUrls] = useState<Record<string, string>>({});
  const [loadingImages, setLoadingImages] = useState<Record<string, boolean>>({});
  const [codeOpen, setCodeOpen] = useState(false);
  const proxyBaseUrl = getProxyBaseUrl();

  useEffect(() => {
    const createdUrls: string[] = [];
    let cancelled = false;

    const fetchImages = async () => {
      for (const annotation of annotations) {
        if (!isImageFilename(annotation.filename) || !annotation.container_id || !annotation.file_id) {
          continue;
        }

        if (!cancelled) {
          setLoadingImages((prev) => ({ ...prev, [annotation.file_id]: true }));
        }

        try {
          const response = await fetch(
            `${proxyBaseUrl}/v1/containers/${annotation.container_id}/files/${annotation.file_id}/content`,
            {
              headers: {
                [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
              },
            },
          );

          if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            createdUrls.push(url);
            if (!cancelled) {
              setImageUrls((prev) => ({ ...prev, [annotation.file_id]: url }));
            } else {
              URL.revokeObjectURL(url);
            }
          }
        } catch (error) {
          console.error("Error fetching image:", error);
        } finally {
          if (!cancelled) {
            setLoadingImages((prev) => ({ ...prev, [annotation.file_id]: false }));
          }
        }
      }
    };

    if (annotations.length > 0 && accessToken) {
      void fetchImages();
    }

    return () => {
      cancelled = true;
      createdUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [annotations, accessToken, proxyBaseUrl]);

  const handleDownload = async (annotation: ContainerFileCitation) => {
    try {
      const response = await fetch(
        `${proxyBaseUrl}/v1/containers/${annotation.container_id}/files/${annotation.file_id}/content`,
        {
          headers: {
            [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          },
        },
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = annotation.filename || `file_${annotation.file_id}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error("Error downloading file:", error);
    }
  };

  const imageAnnotations = annotations.filter((a) => isImageFilename(a.filename));
  const fileAnnotations = annotations.filter((a) => !isImageFilename(a.filename));

  if (!code && annotations.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-3">
      {code && (
        <Collapsible open={codeOpen} onOpenChange={setCodeOpen} className="rounded-md border border-border">
          <CollapsibleTrigger
            render={
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 text-sm text-muted-foreground"
              />
            }
          >
            <Code className="size-4" />
            Python Code Executed
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t border-border p-2">
              <SyntaxHighlighter
                language="python"
                style={syntaxTheme}
                customStyle={{
                  margin: 0,
                  borderRadius: "6px",
                  fontSize: "12px",
                  maxHeight: "300px",
                  overflow: "auto",
                }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {imageAnnotations.map((annotation) => (
        <div key={annotation.file_id} className="overflow-hidden rounded-lg border border-border">
          {loadingImages[annotation.file_id] ? (
            <div className="flex items-center justify-center bg-muted p-8">
              <Loader2 className="size-4 animate-spin text-muted-foreground" aria-hidden="true" />
              <span className="ml-2 text-sm text-muted-foreground">Loading image...</span>
            </div>
          ) : imageUrls[annotation.file_id] ? (
            <div>
              <img
                src={imageUrls[annotation.file_id]}
                alt={annotation.filename || "Generated chart"}
                className="max-h-[400px] max-w-full"
              />
              <div className="flex items-center justify-between border-t border-border bg-muted px-3 py-2">
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <FileImage className="size-3" aria-hidden="true" />
                  {annotation.filename}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  className="h-auto gap-1 px-1 py-0 text-xs text-info hover:text-info/80"
                  onClick={() => void handleDownload(annotation)}
                >
                  <Download className="size-3" />
                  Download
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center bg-muted p-4">
              <span className="text-sm text-muted-foreground">Image not available</span>
            </div>
          )}
        </div>
      ))}

      {fileAnnotations.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {fileAnnotations.map((annotation) => (
            <Button
              key={annotation.file_id}
              type="button"
              variant="outline"
              size="sm"
              className="h-auto gap-2 border-border bg-muted px-3 py-2 hover:bg-accent"
              onClick={() => void handleDownload(annotation)}
            >
              <FileText className="size-4 text-info" aria-hidden="true" />
              <span className="text-sm">{annotation.filename}</span>
              <Download className="size-3 text-muted-foreground" aria-hidden="true" />
            </Button>
          ))}
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterOutput;
