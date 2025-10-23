import React, { useState } from "react";

interface RecognitionMetadata {
  recognizer_name: string;
  recognizer_identifier: string;
}

export interface GuardrailEntity {
  end: number;
  score: number;
  start: number;
  entity_type: string;
  analysis_explanation: string | null;
  recognition_metadata: RecognitionMetadata;
}

interface PresidioDetectedEntitiesProps {
  entities: GuardrailEntity[];
}

const getScoreColor = (score: number): string => {
  if (score >= 0.8) return "text-green-600";
  return "text-yellow-600";
};

const PresidioDetectedEntities = ({ entities }: PresidioDetectedEntitiesProps) => {
  const [entityListExpanded, setEntityListExpanded] = useState(true);
  const [expandedEntities, setExpandedEntities] = useState<Record<number, boolean>>({});

  const toggleEntity = (index: number) => {
    setExpandedEntities((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  if (!entities || entities.length === 0) return null;

  return (
    <div className="mt-4">
      <div
        className="flex items-center mb-2 cursor-pointer"
        onClick={() => setEntityListExpanded(!entityListExpanded)}
      >
        <svg
          className={`w-5 h-5 mr-2 transition-transform ${entityListExpanded ? "transform rotate-90" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <h4 className="font-medium">Detected Entities ({entities.length})</h4>
      </div>

      {entityListExpanded && (
        <div className="space-y-2">
          {entities.map((entity, index) => {
            const isExpanded = expandedEntities[index] || false;

            return (
              <div key={index} className="border rounded-lg overflow-hidden">
                <div
                  className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                  onClick={() => toggleEntity(index)}
                >
                  <div className="flex items-center">
                    <svg
                      className={`w-5 h-5 mr-2 transition-transform ${isExpanded ? "transform rotate-90" : ""}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    <span className="font-medium mr-2">{entity.entity_type}</span>
                    <span className={`font-mono ${getScoreColor(entity.score)}`}>
                      Score: {entity.score.toFixed(2)}
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">Position: {entity.start}-{entity.end}</span>
                </div>

                {isExpanded && (
                  <div className="p-3 border-t bg-white">
                    <div className="grid grid-cols-2 gap-4 mb-2">
                      <div className="space-y-2">
                        <div className="flex">
                          <span className="font-medium w-1/3">Entity Type:</span>
                          <span>{entity.entity_type}</span>
                        </div>
                        <div className="flex">
                          <span className="font-medium w-1/3">Position:</span>
                          <span>Characters {entity.start}-{entity.end}</span>
                        </div>
                        <div className="flex">
                          <span className="font-medium w-1/3">Confidence:</span>
                          <span className={getScoreColor(entity.score)}>{entity.score.toFixed(2)}</span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        {entity.recognition_metadata && (
                          <>
                            <div className="flex">
                              <span className="font-medium w-1/3">Recognizer:</span>
                              <span>{entity.recognition_metadata.recognizer_name}</span>
                            </div>
                            <div className="flex overflow-hidden">
                              <span className="font-medium w-1/3">Identifier:</span>
                              <span className="truncate text-xs font-mono">
                                {entity.recognition_metadata.recognizer_identifier}
                              </span>
                            </div>
                          </>
                        )}
                        {entity.analysis_explanation && (
                          <div className="flex">
                            <span className="font-medium w-1/3">Explanation:</span>
                            <span>{entity.analysis_explanation}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default PresidioDetectedEntities;
