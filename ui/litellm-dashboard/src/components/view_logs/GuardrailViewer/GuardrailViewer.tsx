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
  data: GuardrailInformation | GuardrailInformation[];
}

interface GuardrailDetailsProps {
  entry: GuardrailInformation;
  index: number;
  total: number;
}

const formatTime = (timestamp: number) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
};

const GuardrailDetails = ({ entry, index, total }: GuardrailDetailsProps) => {
  const guardrailProvider = entry.guardrail_provider ?? "presidio";
  const statusLabel = entry.guardrail_status ?? "unknown";
  const isSuccess = statusLabel.toLowerCase() === "success";
  const maskedEntityCount = entry.masked_entity_count || {};
  const totalMaskedEntities = Object.values(maskedEntityCount).reduce(
    (sum, count) => sum + (typeof count === "number" ? count : 0),
    0,
  );

  const guardrailResponse = entry.guardrail_response;
  const presidioEntities = Array.isArray(guardrailResponse) ? guardrailResponse : [];
  const bedrockResponse =
    guardrailProvider === "bedrock" &&
    guardrailResponse !== null &&
    typeof guardrailResponse === "object" &&
    !Array.isArray(guardrailResponse)
      ? (guardrailResponse as BedrockGuardrailResponse)
      : undefined;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {total > 1 && (
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-base font-semibold">
            Guardrail #{index + 1}
            <span className="ml-2 font-mono text-sm text-gray-600">{entry.guardrail_name}</span>
          </h4>
          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded-md text-xs capitalize">
            {guardrailProvider}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex">
            <span className="font-medium w-1/3">Guardrail Name:</span>
            <span className="font-mono break-words">{entry.guardrail_name}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Mode:</span>
            <span className="font-mono break-words">{entry.guardrail_mode}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Status:</span>
            <Tooltip title={isSuccess ? null : "Guardrail failed to run."} placement="top" arrow destroyTooltipOnHide>
              <span
                className={`px-2 py-1 rounded-md text-xs font-medium inline-block ${
                  isSuccess ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800 cursor-help"
                }`}
              >
                {statusLabel}
              </span>
            </Tooltip>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex">
            <span className="font-medium w-1/3">Start Time:</span>
            <span>{formatTime(entry.start_time)}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">End Time:</span>
            <span>{formatTime(entry.end_time)}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Duration:</span>
            <span>{entry.duration.toFixed(4)}s</span>
          </div>
        </div>
      </div>

      {totalMaskedEntities > 0 && (
        <div className="mt-4 pt-4 border-t">
          <h5 className="font-medium mb-2">Masked Entity Summary</h5>
          <div className="flex flex-wrap gap-2">
            {Object.entries(maskedEntityCount).map(([entityType, count]) => (
              <span key={entityType} className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-md text-xs font-medium">
                {entityType}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {guardrailProvider === "presidio" && presidioEntities.length > 0 && (
        <div className="mt-4">
          <PresidioDetectedEntities entities={presidioEntities} />
        </div>
      )}

      {guardrailProvider === "bedrock" && bedrockResponse && (
        <div className="mt-4">
          <BedrockGuardrailDetails response={bedrockResponse} />
        </div>
      )}
    </div>
  );
};

const GuardrailViewer = ({ data }: GuardrailViewerProps) => {
  const guardrailEntries = Array.isArray(data)
    ? data.filter((entry): entry is GuardrailInformation => Boolean(entry))
    : data
      ? [data]
      : [];

  const [sectionExpanded, setSectionExpanded] = useState(true);

  const primaryName =
    guardrailEntries.length === 1 ? guardrailEntries[0].guardrail_name : `${guardrailEntries.length} guardrails`;
  const statuses = Array.from(new Set(guardrailEntries.map((entry) => entry.guardrail_status)));
  const allSucceeded = statuses.every((status) => (status ?? "").toLowerCase() === "success");
  const aggregatedStatus = allSucceeded ? "success" : "failure";
  const totalMaskedEntities = guardrailEntries.reduce((sum, entry) => {
    return (
      sum +
      Object.values(entry.masked_entity_count || {}).reduce(
        (acc, count) => acc + (typeof count === "number" ? count : 0),
        0,
      )
    );
  }, 0);

  const tooltipTitle = allSucceeded ? null : "Guardrail failed to run.";

  if (guardrailEntries.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow mb-6">
      <div
        className="flex justify-between items-center p-4 border-b cursor-pointer hover:bg-gray-50"
        onClick={() => setSectionExpanded(!sectionExpanded)}
      >
        <div className="flex items-center gap-2">
          <svg
            className={`w-5 h-5 text-gray-600 transition-transform ${sectionExpanded ? "transform rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h3 className="text-lg font-medium">Guardrail Information</h3>

          <Tooltip title={tooltipTitle} placement="top" arrow destroyTooltipOnHide>
            <span
              className={`ml-2 px-2 py-1 rounded-md text-xs font-medium inline-block ${
                allSucceeded ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800 cursor-help"
              }`}
            >
              {aggregatedStatus}
            </span>
          </Tooltip>

          <span className="ml-2 font-mono text-sm text-gray-600">{primaryName}</span>

          {totalMaskedEntities > 0 && (
            <span className="ml-2 px-2 py-1 bg-blue-50 text-blue-700 rounded-md text-xs font-medium">
              {totalMaskedEntities} masked {totalMaskedEntities === 1 ? "entity" : "entities"}
            </span>
          )}
        </div>
        <span className="text-sm text-gray-500">{sectionExpanded ? "Click to collapse" : "Click to expand"}</span>
      </div>

      {sectionExpanded && (
        <div className="p-4 space-y-6">
          {guardrailEntries.map((entry, index) => (
            <GuardrailDetails
              key={`${entry.guardrail_name ?? "guardrail"}-${index}`}
              entry={entry}
              index={index}
              total={guardrailEntries.length}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default GuardrailViewer;
