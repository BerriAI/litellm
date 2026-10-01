export interface SystemOneChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
}

export interface SystemOneNoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: {
    true?: string;
    false?: string;
  };
}

export interface SystemOneScoreQuestion {
  type: "score";
  instructions: string;
  criteria: string[];
}

export type SystemOneQuestion = SystemOneChoiceQuestion | SystemOneNoulQuestion | SystemOneScoreQuestion;

export interface SystemOneRequest {
  model?: string;
  state: unknown;
  questions: Record<string, SystemOneQuestion>;
}

export interface SystemOneNoulAnswer {
  type: "noul";
  noul: number;
}

export interface SystemOneChoiceAnswer {
  type: "choice";
  choice: string;
  confidence?: number;
  probabilities: Record<string, number>;
}

export interface SystemOneScoreAnswer {
  type: "score";
  score: number;
  confidence?: number;
  legend?: Record<string, string>;
  probabilities: Record<string, number>;
}

export type SystemOneAnswer = SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer;

export interface SystemOneResponse {
  model: string;
  answers: Record<string, SystemOneAnswer>;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}
