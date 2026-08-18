export type LlmOutputErrorCode =
  | "empty_output"
  | "truncated_output"
  | "invalid_json"
  | "invalid_schema";

/** A transport-success response whose generated content is unusable. */
export class LlmOutputError extends Error {
  readonly code: LlmOutputErrorCode;

  constructor(code: LlmOutputErrorCode, message: string) {
    super(message);
    this.name = "LlmOutputError";
    this.code = code;
  }
}

export function isLlmOutputError(error: unknown): error is LlmOutputError {
  return error instanceof LlmOutputError;
}
