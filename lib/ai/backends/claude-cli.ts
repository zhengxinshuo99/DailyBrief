import { spawn } from "node:child_process";
import path from "node:path";
import { LlmOutputError } from "../errors";
import { classifyError, logLlmCall } from "../log";
import type { LlmRunOptions, LlmRunResult } from "../llm";

export const CLAUDE_MODEL = process.env.CLAUDE_MODEL?.trim() || "sonnet";

function resolveCliPath(): string {
  const override = process.env.CLAUDE_CLI_PATH?.trim();
  if (override) return override;
  const appdata = process.env.APPDATA;
  if (appdata) return path.join(appdata, "npm", "claude.cmd");
  return "claude";
}

/**
 * Invoke the local `claude` CLI in print mode against the Max subscription.
 * Writes the user prompt over stdin to bypass shell argument length limits.
 *
 * stderr is logged as warnings but not thrown — plugins like claude-mem
 * sometimes emit non-fatal hook errors on stderr that the CLI itself
 * still completes around.
 */
export function runClaudeCli({
  systemPrompt,
  userPrompt,
  timeoutMs = 180_000,
}: LlmRunOptions): Promise<LlmRunResult> {
  const cli = resolveCliPath();
  const args = [
    "--print",
    "--model",
    CLAUDE_MODEL,
    "--append-system-prompt",
    systemPrompt,
  ];
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const child = spawn(cli, args, {
      shell: true,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (err: Error | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const durationMs = Date.now() - started;
      const finalErr =
        err ??
        (stdout.trim()
          ? null
          : new LlmOutputError(
              "empty_output",
              "claude CLI returned empty completion content",
            ));
      const success = finalErr === null;
      logLlmCall({
        ts: new Date(started).toISOString(),
        backend: "claude-cli",
        model: CLAUDE_MODEL,
        durationMs,
        success,
        inputChars: systemPrompt.length + userPrompt.length,
        outputChars: stdout.length,
        errorCategory: success
          ? null
          : classifyError(`${stderr}\n${finalErr?.message ?? ""}`),
        errorSnippet:
          !success && stderr.trim() ? stderr.trim().slice(0, 200) : null,
      });
      if (finalErr) reject(finalErr);
      else resolve({ text: stdout.trim(), durationMs });
    };

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      finish(new Error(`claude CLI timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (err) => finish(err));
    child.on("close", (code) => {
      if (stderr.trim()) {
        console.warn(`[claude-cli] stderr (non-fatal): ${stderr.trim()}`);
      }
      if (code !== 0 && !stdout.trim()) {
        finish(new Error(`claude CLI exited ${code} with empty stdout`));
        return;
      }
      finish(null);
    });

    child.stdin.write(userPrompt);
    child.stdin.end();
  });
}
