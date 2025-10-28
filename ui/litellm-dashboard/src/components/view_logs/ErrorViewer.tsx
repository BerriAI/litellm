import React from "react";

interface ErrorViewerProps {
  errorInfo: {
    error_class?: string;
    error_message?: string;
    traceback?: string;
    llm_provider?: string;
    error_code?: string | number;
  };
}

export const ErrorViewer: React.FC<ErrorViewerProps> = ({ errorInfo }) => {
  const [expandedFrames, setExpandedFrames] = React.useState<{ [key: number]: boolean }>({});
  const [allExpanded, setAllExpanded] = React.useState(false);

  // Toggle individual frame
  const toggleFrame = (index: number) => {
    setExpandedFrames((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // Toggle all frames
  const toggleAllFrames = () => {
    const newState = !allExpanded;
    setAllExpanded(newState);

    if (tracebackFrames.length > 0) {
      const newExpandedState: { [key: number]: boolean } = {};
      tracebackFrames.forEach((_, idx) => {
        newExpandedState[idx] = newState;
      });
      setExpandedFrames(newExpandedState);
    }
  };

  // Parse traceback into frames
  const parseTraceback = (traceback: string) => {
    if (!traceback) return [];

    // Extract file paths, line numbers and code from traceback
    const fileLineRegex = /File "([^"]+)", line (\d+)/g;
    const matches = Array.from(traceback.matchAll(fileLineRegex));

    // Create simplified frames
    return matches.map((match) => {
      const filePath = match[1];
      const lineNumber = match[2];
      const fileName = filePath.split("/").pop() || filePath;

      // Extract the context around this frame
      const matchIndex = match.index || 0;
      const nextMatchIndex = traceback.indexOf('File "', matchIndex + 1);
      const frameContent =
        nextMatchIndex > -1
          ? traceback.substring(matchIndex, nextMatchIndex).trim()
          : traceback.substring(matchIndex).trim();

      // Try to extract the code line
      const lines = frameContent.split("\n");
      let code = "";
      if (lines.length > 1) {
        code = lines[lines.length - 1].trim();
      }

      return {
        filePath,
        fileName,
        lineNumber,
        code,
        inFunction: frameContent.includes(" in ") ? frameContent.split(" in ")[1].split("\n")[0] : "",
      };
    });
  };

  const tracebackFrames = errorInfo.traceback ? parseTraceback(errorInfo.traceback) : [];

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-4 border-b">
        <h3 className="text-lg font-medium flex items-center text-red-600">
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          Error Details
        </h3>
      </div>

      <div className="p-4">
        <div className="bg-red-50 rounded-md p-4 mb-4">
          <div className="flex">
            <span className="text-red-800 font-medium w-20">Type:</span>
            <span className="text-red-700">{errorInfo.error_class || "Unknown Error"}</span>
          </div>
          <div className="flex mt-2">
            <span className="text-red-800 font-medium w-20 flex-shrink-0">Message:</span>
            <span className="text-red-700 break-words whitespace-pre-wrap">
              {errorInfo.error_message || "Unknown error occurred"}
            </span>
          </div>
        </div>

        {errorInfo.traceback && (
          <div className="mt-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-medium">Traceback</h4>
              <div className="flex items-center space-x-4">
                <button
                  onClick={toggleAllFrames}
                  className="text-gray-500 hover:text-gray-700 flex items-center text-sm"
                >
                  {allExpanded ? "Collapse All" : "Expand All"}
                </button>
                <button
                  onClick={() => navigator.clipboard.writeText(errorInfo.traceback || "")}
                  className="text-gray-500 hover:text-gray-700 flex items-center"
                  title="Copy traceback"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span className="ml-1">Copy</span>
                </button>
              </div>
            </div>

            <div className="bg-white rounded-md border border-gray-200 overflow-hidden shadow-sm">
              {tracebackFrames.map((frame, index) => (
                <div key={index} className="border-b border-gray-200 last:border-b-0">
                  <div
                    className="px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-gray-50"
                    onClick={() => toggleFrame(index)}
                  >
                    <div className="flex items-center">
                      <span className="text-gray-400 mr-2 w-12 text-right">{frame.lineNumber}</span>
                      <span className="text-gray-600 font-medium">{frame.fileName}</span>
                      <span className="text-gray-500 mx-1">in</span>
                      <span className="text-indigo-600 font-medium">{frame.inFunction || frame.fileName}</span>
                    </div>
                    <svg
                      className={`w-5 h-5 text-gray-500 transition-transform ${expandedFrames[index] ? "transform rotate-180" : ""}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                  {(expandedFrames[index] || false) && frame.code && (
                    <div className="px-12 py-2 font-mono text-sm text-gray-800 bg-gray-50 overflow-x-auto border-t border-gray-100">
                      {frame.code}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
