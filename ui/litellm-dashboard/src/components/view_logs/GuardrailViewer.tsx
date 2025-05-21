import React, { useState } from "react";

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
  guardrail_response: GuardrailEntity[];
  masked_entity_count: MaskedEntityCount;
}

interface GuardrailViewerProps {
  data: GuardrailInformation;
}

export function GuardrailViewer({ data }: GuardrailViewerProps) {
  const [sectionExpanded, setSectionExpanded] = useState(true);
  const [entityListExpanded, setEntityListExpanded] = useState(true);
  const [expandedEntities, setExpandedEntities] = useState<Record<number, boolean>>({});

  if (!data) {
    return null;
  }

  // Calculate total masked entities
  const totalMaskedEntities = data.masked_entity_count ? 
    Object.values(data.masked_entity_count).reduce((sum, count) => sum + count, 0) : 0;

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const toggleEntity = (index: number) => {
    setExpandedEntities(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return "text-green-600";
    return "text-yellow-600";
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
          <span className={`ml-3 px-2 py-1 rounded-md text-xs font-medium inline-block ${
            data.guardrail_status === "success" 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {data.guardrail_status}
          </span>
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
                  <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block ${
                    data.guardrail_status === "success" 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {data.guardrail_status}
                  </span>
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

          {/* Detected Entities Section */}
          {data.guardrail_response && data.guardrail_response.length > 0 && (
            <div className="mt-4">
              <div 
                className="flex items-center mb-2 cursor-pointer"
                onClick={() => setEntityListExpanded(!entityListExpanded)}
              >
                <svg 
                  className={`w-5 h-5 mr-2 transition-transform ${entityListExpanded ? 'transform rotate-90' : ''}`}
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <h4 className="font-medium">Detected Entities ({data.guardrail_response.length})</h4>
              </div>
              
              {entityListExpanded && (
                <div className="space-y-2">
                  {data.guardrail_response.map((entity, index) => {
                    const isExpanded = expandedEntities[index] || false;
                    
                    return (
                      <div key={index} className="border rounded-lg overflow-hidden">
                        <div 
                          className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                          onClick={() => toggleEntity(index)}
                        >
                          <div className="flex items-center">
                            <svg 
                              className={`w-5 h-5 mr-2 transition-transform ${isExpanded ? 'transform rotate-90' : ''}`}
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
                          <span className="text-xs text-gray-500">
                            Position: {entity.start}-{entity.end}
                          </span>
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
                                  <span className={getScoreColor(entity.score)}>
                                    {entity.score.toFixed(2)}
                                  </span>
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
          )}
        </div>
      )}
    </div>
  );
} 