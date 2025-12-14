/**
 * Code Interpreter event handling for the Responses API.
 */

export interface CodeInterpreterResult {
  code: string;
  containerId: string;
  annotations: Array<{
    type: "container_file_citation";
    container_id: string;
    file_id: string;
    filename: string;
    start_index: number;
    end_index: number;
  }>;
}

export interface CodeInterpreterState {
  code: string;
  containerId: string;
}

/**
 * Handle code interpreter call completed event.
 * Extracts code and container ID from the event.
 */
export function handleCodeInterpreterCall(
  event: any,
  state: CodeInterpreterState
): CodeInterpreterState {
  if (event.type === "response.output_item.done" && event.item?.type === "code_interpreter_call") {
    console.log("Code interpreter call completed:", event.item);
    return {
      code: event.item.code || "",
      containerId: event.item.container_id || "",
    };
  }
  return state;
}

/**
 * Handle code interpreter output with file annotations.
 * Calls the callback if file annotations are present.
 */
export function handleCodeInterpreterOutput(
  event: any,
  state: CodeInterpreterState,
  onCodeInterpreterResult?: (result: CodeInterpreterResult) => void
): void {
  if (
    event.type === "response.output_item.done" &&
    event.item?.type === "message" &&
    event.item?.content &&
    onCodeInterpreterResult
  ) {
    const content = event.item.content;
    for (const part of content) {
      if (part.type === "output_text" && part.annotations) {
        const fileAnnotations = part.annotations.filter(
          (a: any) => a.type === "container_file_citation"
        );
        if (fileAnnotations.length > 0 || state.code) {
          onCodeInterpreterResult({
            code: state.code,
            containerId: state.containerId,
            annotations: fileAnnotations,
          });
        }
      }
    }
  }
}

/**
 * Check if code interpreter is being used based on event type.
 */
export function isCodeInterpreterEvent(event: any): boolean {
  return (
    event.type === "response.output_item.done" &&
    event.item?.type === "code_interpreter_call"
  );
}

