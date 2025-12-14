import React, { useState, useEffect } from "react";
import { Collapse, Spin } from "antd";
import {
  CodeOutlined,
  DownloadOutlined,
  FileImageOutlined,
  FileTextOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { getProxyBaseUrl } from "@/components/networking";

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
  containerId,
  annotations = [],
  accessToken,
}) => {
  const [imageUrls, setImageUrls] = useState<Record<string, string>>({});
  const [loadingImages, setLoadingImages] = useState<Record<string, boolean>>({});
  const proxyBaseUrl = getProxyBaseUrl();

  // Fetch images from container files API
  useEffect(() => {
    const fetchImages = async () => {
      for (const annotation of annotations) {
        const isImage = annotation.filename?.toLowerCase().endsWith(".png") ||
                       annotation.filename?.toLowerCase().endsWith(".jpg") ||
                       annotation.filename?.toLowerCase().endsWith(".jpeg") ||
                       annotation.filename?.toLowerCase().endsWith(".gif");
        
        if (isImage && annotation.container_id && annotation.file_id) {
          setLoadingImages(prev => ({ ...prev, [annotation.file_id]: true }));
          
          try {
            // Fetch image content from container files API
            const response = await fetch(
              `${proxyBaseUrl}/v1/containers/${annotation.container_id}/files/${annotation.file_id}/content`,
              {
                headers: {
                  Authorization: `Bearer ${accessToken}`,
                },
              }
            );
            
            if (response.ok) {
              const blob = await response.blob();
              const url = URL.createObjectURL(blob);
              setImageUrls(prev => ({ ...prev, [annotation.file_id]: url }));
            }
          } catch (error) {
            console.error("Error fetching image:", error);
          } finally {
            setLoadingImages(prev => ({ ...prev, [annotation.file_id]: false }));
          }
        }
      }
    };

    if (annotations.length > 0 && accessToken) {
      fetchImages();
    }

    // Cleanup URLs on unmount
    return () => {
      Object.values(imageUrls).forEach(url => URL.revokeObjectURL(url));
    };
  }, [annotations, accessToken, proxyBaseUrl]);

  const handleDownload = async (annotation: ContainerFileCitation) => {
    try {
      const response = await fetch(
        `${proxyBaseUrl}/v1/containers/${annotation.container_id}/files/${annotation.file_id}/content`,
        {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
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

  // Separate images and other files
  const imageAnnotations = annotations.filter(a => 
    a.filename?.toLowerCase().endsWith(".png") ||
    a.filename?.toLowerCase().endsWith(".jpg") ||
    a.filename?.toLowerCase().endsWith(".jpeg") ||
    a.filename?.toLowerCase().endsWith(".gif")
  );
  
  const fileAnnotations = annotations.filter(a => 
    !a.filename?.toLowerCase().endsWith(".png") &&
    !a.filename?.toLowerCase().endsWith(".jpg") &&
    !a.filename?.toLowerCase().endsWith(".jpeg") &&
    !a.filename?.toLowerCase().endsWith(".gif")
  );

  if (!code && annotations.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-3">
      {/* Executed Code - Collapsible */}
      {code && (
        <Collapse
          size="small"
          items={[
            {
              key: "code",
              label: (
                <span className="flex items-center gap-2 text-sm text-gray-600">
                  <CodeOutlined /> Python Code Executed
                </span>
              ),
              children: (
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
              ),
            },
          ]}
        />
      )}

      {/* Generated Images */}
      {imageAnnotations.map((annotation) => (
        <div key={annotation.file_id} className="rounded-lg border border-gray-200 overflow-hidden">
          {loadingImages[annotation.file_id] ? (
            <div className="flex items-center justify-center p-8 bg-gray-50">
              <Spin indicator={<LoadingOutlined spin />} />
              <span className="ml-2 text-sm text-gray-500">Loading image...</span>
            </div>
          ) : imageUrls[annotation.file_id] ? (
            <div>
              <img
                src={imageUrls[annotation.file_id]}
                alt={annotation.filename || "Generated chart"}
                className="max-w-full"
                style={{ maxHeight: "400px" }}
              />
              <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-t border-gray-200">
                <span className="text-xs text-gray-500 flex items-center gap-1">
                  <FileImageOutlined /> {annotation.filename}
                </span>
                <button
                  onClick={() => handleDownload(annotation)}
                  className="text-xs text-blue-500 hover:text-blue-700 flex items-center gap-1"
                >
                  <DownloadOutlined /> Download
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center p-4 bg-gray-50">
              <span className="text-sm text-gray-400">Image not available</span>
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
              className="flex items-center gap-2 px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <FileTextOutlined className="text-blue-500" />
              <span className="text-sm">{annotation.filename}</span>
              <DownloadOutlined className="text-gray-400" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterOutput;

