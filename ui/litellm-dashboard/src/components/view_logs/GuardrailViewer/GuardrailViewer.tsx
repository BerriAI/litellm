import React, { useState } from "react";
import { Tooltip } from "antd";
import PresidioDetectedEntities from "./PresidioDetectedEntities";
import BedrockGuardrailDetails, {
  BedrockGuardrailResponse,
} from "@/components/view_logs/GuardrailViewer/BedrockGuardrailDetails";

interface RecognitionMetadata {
  recognizer_name: string;
  recognizer_identifier: string;
}

interface GuardrailEntity {
  end: number;
  score: number;
  start: number;
  entity_type: string;
  analysis_explanation: string | null;
  recognition_metadata: RecognitionMetadata;
}

interface MaskedEntityCount {
  [key: string]: number;
}

interface GuardrailInformation {
  duration: number;
  end_time: number;
  start_time: number;
  guardrail_mode: string;
  guardrail_name: string;
  guardrail_status: string;
  guardrail_response: GuardrailEntity[] | BedrockGuardrailResponse;
  masked_entity_count: MaskedEntityCount;
  guardrail_provider?: string; // "presidio" | other providers
}

interface GuardrailViewerProps {
  data: GuardrailInformation;
}

const GuardrailViewer = ({ data }: GuardrailViewerProps) => {
  const [sectionExpanded, setSectionExpanded] = useState(true);

  // Default to presidio for backwards compatibility
  const guardrailProvider = data.guardrail_provider ?? "presidio";

  if (!data) return null;

  const isSuccess =
    typeof data.guardrail_status === "string" &&
    data.guardrail_status.toLowerCase() === "success";

  const tooltipTitle = isSuccess ? null : "Guardrail failed to run.";

  // Calculate total masked entities
  const totalMaskedEntities = data.masked_entity_count ? 
    Object.values(data.masked_entity_count).reduce((sum, count) => sum + count, 0) : 0;

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  return (
    <div className="bg-white rounded-lg shadow mb-6">
      <div
        className="flex justify-between items-center p-4 border-b cursor-pointer hover:bg-gray-50"
        onClick={() => setSectionExpanded(!sectionExpanded)}
      >
        <div className="flex items-center">
          <svg 
            className={`w-5 h-5 mr-2 text-gray-600 transition-transform ${sectionExpanded ? 'transform rotate-90' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h3 className="text-lg font-medium">Guardrail Information</h3>

          {/* Header status chip with tooltip */}
          <Tooltip title={tooltipTitle} placement="top" arrow destroyTooltipOnHide>
            <span
              className={`ml-3 px-2 py-1 rounded-md text-xs font-medium inline-block ${
                isSuccess ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800 cursor-help"
              }`}
            >
              {data.guardrail_status}
            </span>
          </Tooltip>

          {totalMaskedEntities > 0 && (
            <span className="ml-3 px-2 py-1 bg-blue-50 text-blue-700 rounded-md text-xs font-medium">
              {totalMaskedEntities} masked {totalMaskedEntities === 1 ? 'entity' : 'entities'}
            </span>
          )}
        </div>
        <span className="text-sm text-gray-500">{sectionExpanded ? 'Click to collapse' : 'Click to expand'}</span>
      </div>

      {sectionExpanded && (
        <div className="p-4">
          <div className="bg-white rounded-lg border p-4 mb-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex">
                  <span className="font-medium w-1/3">Guardrail Name:</span>
                  <span className="font-mono">{data.guardrail_name}</span>
                </div>
                <div className="flex">
                  <span className="font-medium w-1/3">Mode:</span>
                  <span className="font-mono">{data.guardrail_mode}</span>
                </div>
                <div className="flex">
                  <span className="font-medium w-1/3">Status:</span>
                  <Tooltip title={tooltipTitle} placement="top" arrow destroyTooltipOnHide>
                    <span
                      className={`px-2 py-1 rounded-md text-xs font-medium inline-block ${
                        isSuccess ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800 cursor-help"
                      }`}
                    >
                      {data.guardrail_status}
                    </span>
                  </Tooltip>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex">
                  <span className="font-medium w-1/3">Start Time:</span>
                  <span>{formatTime(data.start_time)}</span>
                </div>
                <div className="flex">
                  <span className="font-medium w-1/3">End Time:</span>
                  <span>{formatTime(data.end_time)}</span>
                </div>
                <div className="flex">
                  <span className="font-medium w-1/3">Duration:</span>
                  <span>{data.duration.toFixed(4)}s</span>
                </div>
              </div>
            </div>

            {/* Masked Entity Summary */}
            {data.masked_entity_count && Object.keys(data.masked_entity_count).length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="font-medium mb-2">Masked Entity Summary</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(data.masked_entity_count).map(([entityType, count]) => (
                    <span key={entityType} className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-md text-xs font-medium">
                      {entityType}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Provider-specific Detected Entities */}
          {guardrailProvider === "presidio" && (data.guardrail_response as GuardrailEntity[])?.length > 0 && (
            <PresidioDetectedEntities entities={data.guardrail_response as GuardrailEntity[]} />
          )}

          {guardrailProvider === "bedrock" && data.guardrail_response && (
            <div className="mt-4">
              <BedrockGuardrailDetails response={data.guardrail_response as BedrockGuardrailResponse} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GuardrailViewer;
