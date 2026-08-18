import OpenAI from "openai";
import { LlmOutputError } from "../errors";
import { classifyError, logLlmCall } from "../log";
import type { LlmRunOptions, LlmRunResult } from "../llm";

/**
 * OpenAI-compatible backend. Reused for any provider that exposes the
 * standard `/chat/completions` endpoint: OpenAI itself, DeepSeek, MiniMax,
 * Groq, Together, OpenRouter, local LM Studio / Ollama, etc.
 */
export interface OpenAICompatConfig {
  /** Stable backend id, used in logs and error messages */
  backend: "openai" | "deepseek" | "minimax";
  defaultBaseUrl: string;
  defaultModel: string;
  apiKeyEnv: string;
  baseUrlEnv: string;
}

export const PRESETS: Record<OpenAICompatConfig["backend"], OpenAICompatConfig> = {
  openai: {
    backend: "openai",
    defaultBaseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    apiKeyEnv: "OPENAI_API_KEY",
    baseUrlEnv: "OPENAI_BASE_URL",
  },
  deepseek: {
    backend: "deepseek",
    defaultBaseUrl: "https://api.deepseek.com/v1",
    // deepseek-chat alias retires 2026-07-24 — point new users at the
    // current production model instead.
    defaultModel: "deepseek-v4-flash",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    baseUrlEnv: "DEEPSEEK_BASE_URL",
  },
  minimax: {
    backend: "minimax",
    defaultBaseUrl: "https://api.minimax.io/v1",
    defaultModel: "MiniMax-M2.7",
    apiKeyEnv: "MINIMAX_API_KEY",
    baseUrlEnv: "MINIMAX_BASE_URL",
  },
};

const clientCache = new Map<string, OpenAI>();

function getClient(cfg: OpenAICompatConfig): { client: OpenAI; model: string } {
  // Provider-specific env wins; LLM_API_KEY / LLM_BASE_URL are generic
  // aliases so users pointing at a non-preset OpenAI-compatible service
  // (Moonshot, SiliconFlow, OpenRouter, self-hosted vLLM, ...) don't have
  // to misuse the OPENAI_* variable names just to reach a custom endpoint.
  const apiKey = process.env[cfg.apiKeyEnv] || process.env.LLM_API_KEY;
  if (!apiKey) {
    throw new Error(
      `${cfg.apiKeyEnv} (or generic LLM_API_KEY) is required for LLM_BACKEND=${cfg.backend}. Set it in .env.local.`,
    );
  }
  const baseURL = process.env[cfg.baseUrlEnv]?.trim()
    || process.env.LLM_BASE_URL?.trim()
    || cfg.defaultBaseUrl;
  const model = process.env.LLM_MODEL?.trim() || cfg.defaultModel;

  const cacheKey = `${baseURL}::${apiKey.slice(-6)}`;
  let client = clientCache.get(cacheKey);
  if (!client) {
    client = new OpenAI({ apiKey, baseURL });
    clientCache.set(cacheKey, client);
  }
  return { client, model };
}

export function openaiCompatModel(cfg: OpenAICompatConfig): string {
  return process.env.LLM_MODEL?.trim() || cfg.defaultModel;
}

export async function runOpenAICompat(
  opts: LlmRunOptions,
  cfg: OpenAICompatConfig,
): Promise<LlmRunResult> {
  const { client, model } = getClient(cfg);
  const started = Date.now();
  const inputChars = opts.systemPrompt.length + opts.userPrompt.length;
  const timeoutMs = opts.timeoutMs ?? 180_000;
  let outputChars = 0;
  let finishReason: string | null = null;
  let responseId: string | null = null;
  let promptTokens: number | null = null;
  let completionTokens: number | null = null;
  let totalTokens: number | null = null;

  try {
    const request = {
      model,
      messages: [
        { role: "system", content: opts.systemPrompt },
        { role: "user", content: opts.userPrompt },
      ],
      // Explicit max_tokens — most providers default low (DeepSeek 4096,
      // some MiniMax variants 2048). A 16-item batch enrichment routinely
      // exceeds 4K output tokens once you count Chinese chars + JSON
      // structure, and silent truncation made it through with just 1/16
      // entries parseable. 8192 covers all observed daily batches with
      // generous headroom. Match the explicit value Anthropic SDK uses.
      max_tokens: 8192,
      // DeepSeek documents JSON Output support and its prompts in this
      // project already contain the required explicit JSON instruction.
      // Keep other compatibility providers untouched because support for
      // response_format is not universal.
      ...(cfg.backend === "deepseek"
        ? {
            response_format: { type: "json_object" as const },
            // V4 Flash enables high-effort thinking by default. For these
            // structured editorial calls it can consume the output budget
            // before emitting JSON, producing the observed empty content.
            thinking: { type: "disabled" as const },
          }
        : {}),
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming & {
      thinking?: { type: "disabled" };
    };
    const resp = await client.chat.completions.create(request, {
      timeout: timeoutMs,
    });
    responseId = resp.id ?? null;
    promptTokens = resp.usage?.prompt_tokens ?? null;
    completionTokens = resp.usage?.completion_tokens ?? null;
    totalTokens = resp.usage?.total_tokens ?? null;
    const choice = resp.choices[0];
    finishReason = choice?.finish_reason ?? null;
    const text = (choice?.message?.content ?? "").trim();
    outputChars = text.length;

    if (!choice) {
      throw new LlmOutputError(
        "empty_output",
        `${cfg.backend} returned no completion choices`,
      );
    }
    if (
      finishReason === "length" ||
      finishReason === "content_filter" ||
      finishReason === "insufficient_system_resource"
    ) {
      throw new LlmOutputError(
        "truncated_output",
        `${cfg.backend} response was incomplete (finish_reason=${finishReason}, ${outputChars} chars)`,
      );
    }
    if (!text) {
      throw new LlmOutputError(
        "empty_output",
        `${cfg.backend} returned empty completion content (finish_reason=${finishReason ?? "missing"})`,
      );
    }

    const durationMs = Date.now() - started;
    logLlmCall({
      ts: new Date(started).toISOString(),
      backend: cfg.backend,
      model,
      durationMs,
      success: true,
      inputChars,
      outputChars,
      errorCategory: null,
      errorSnippet: null,
      finishReason,
      responseId,
      promptTokens,
      completionTokens,
      totalTokens,
    });
    return { text, durationMs };
  } catch (err) {
    const durationMs = Date.now() - started;
    const msg = err instanceof Error ? err.message : String(err);
    logLlmCall({
      ts: new Date(started).toISOString(),
      backend: cfg.backend,
      model,
      durationMs,
      success: false,
      inputChars,
      outputChars,
      errorCategory: classifyError(msg),
      errorSnippet: msg.slice(0, 200),
      finishReason,
      responseId,
      promptTokens,
      completionTokens,
      totalTokens,
    });
    throw err;
  }
}
