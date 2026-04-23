import React, { useState, useEffect } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Code,
  Download,
  FileImage,
  FileText,
  Loader2,
} from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
} from "@/components/networking";

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

const CodeInterpreterOutput: React.FC<CodeInterpreterOutputProps> = ({
  code,
  annotations = [],
  accessToken,
}) => {
  const [imageUrls, setImageUrls] = useState<Record<string, string>>({});
  const [loadingImages, setLoadingImages] = useState<Record<string, boolean>>(
    {},
  );
  const proxyBaseUrl = getProxyBaseUrl();

  useEffect(() => {
    const fetchImages = async () => {
      for (const annotation of annotations) {
        const isImage =
          annotation.filename?.toLowerCase().endsWith(".png") ||
          annotation.filename?.toLowerCase().endsWith(".jpg") ||
          annotation.filename?.toLowerCase().endsWith(".jpeg") ||
          annotation.filename?.toLowerCase().endsWith(".gif");

        if (isImage && annotation.container_id && annotation.file_id) {
          setLoadingImages((prev) => ({ ...prev, [annotation.file_id]: true }));

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
              setImageUrls((prev) => ({ ...prev, [annotation.file_id]: url }));
            }
          } catch (error) {
            console.error("Error fetching image:", error);
          } finally {
            setLoadingImages((prev) => ({
              ...prev,
              [annotation.file_id]: false,
            }));
          }
        }
      }
    };

    if (annotations.length > 0 && accessToken) {
      fetchImages();
    }

    return () => {
      Object.values(imageUrls).forEach((url) => URL.revokeObjectURL(url));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const imageAnnotations = annotations.filter(
    (a) =>
      a.filename?.toLowerCase().endsWith(".png") ||
      a.filename?.toLowerCase().endsWith(".jpg") ||
      a.filename?.toLowerCase().endsWith(".jpeg") ||
      a.filename?.toLowerCase().endsWith(".gif"),
  );

  const fileAnnotations = annotations.filter(
    (a) =>
      !a.filename?.toLowerCase().endsWith(".png") &&
      !a.filename?.toLowerCase().endsWith(".jpg") &&
      !a.filename?.toLowerCase().endsWith(".jpeg") &&
      !a.filename?.toLowerCase().endsWith(".gif"),
  );

  if (!code && annotations.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-3">
      {/* Executed Code - Collapsible */}
      {code && (
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="code" className="border border-border rounded-md px-3">
            <AccordionTrigger className="py-2 hover:no-underline">
              <span className="flex items-center gap-2 text-sm text-muted-foreground">
                <Code className="h-4 w-4" /> Python Code Executed
              </span>
            </AccordionTrigger>
            <AccordionContent>
              <SyntaxHighlighter
                language="python"
                style={coy}
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
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Generated Images */}
      {imageAnnotations.map((annotation) => (
        <div
          key={annotation.file_id}
          className="rounded-lg border border-border overflow-hidden"
        >
          {loadingImages[annotation.file_id] ? (
            <div className="flex items-center justify-center p-8 bg-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="ml-2 text-sm text-muted-foreground">
                Loading image...
              </span>
            </div>
          ) : imageUrls[annotation.file_id] ? (
            <div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrls[annotation.file_id]}
                alt={annotation.filename || "Generated chart"}
                className="max-w-full"
                style={{ maxHeight: "400px" }}
              />
              <div className="flex items-center justify-between px-3 py-2 bg-muted border-t border-border">
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <FileImage className="h-3.5 w-3.5" /> {annotation.filename}
                </span>
                <button
                  onClick={() => handleDownload(annotation)}
                  className="text-xs text-primary hover:text-primary/80 flex items-center gap-1"
                >
                  <Download className="h-3.5 w-3.5" /> Download
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center p-4 bg-muted">
              <span className="text-sm text-muted-foreground">
                Image not available
              </span>
            </div>
          )}
        </div>
      ))}

      {/* Download Links for Other Files */}
      {fileAnnotations.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {fileAnnotations.map((annotation) => (
            <button
              key={annotation.file_id}
              onClick={() => handleDownload(annotation)}
              className="flex items-center gap-2 px-3 py-2 bg-muted border border-border rounded-lg hover:bg-muted/70 transition-colors"
            >
              <FileText className="h-4 w-4 text-primary" />
              <span className="text-sm">{annotation.filename}</span>
              <Download className="h-4 w-4 text-muted-foreground" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterOutput;
