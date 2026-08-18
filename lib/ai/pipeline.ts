import { jsonrepair } from "jsonrepair";
import { runLlm, type LlmRunner } from "./llm";
import { isLlmOutputError, LlmOutputError } from "./errors";
import { extractJson } from "./json-util";
import { SYSTEM_PROMPT_DIGEST_EN, SYSTEM_PROMPT_DIGEST_ZH } from "./prompts";
import { REPORT_LOCALE } from "../sources/registry";
import type { Category, RawArticle } from "../sources/types";

const SYSTEM_PROMPT_DIGEST =
  REPORT_LOCALE === "en" ? SYSTEM_PROMPT_DIGEST_EN : SYSTEM_PROMPT_DIGEST_ZH;

export interface BriefItem {
  title: string;
  url: string;
  source: string;
  summary: string;
  importance: number;
}

export interface DailyReport {
  hero_headline: string;
  daily_overview: string;
  tech_briefs: BriefItem[];
  finance_briefs: BriefItem[];
  politics_briefs: BriefItem[];
  editor_note: string;
  keywords: string[];
  /** Optional trading-signals section, present when scripts/daily.ts ran successfully. */
  trading?: TradingSection;
}

import type { TickerAnalysis } from "../trading/signals";
import type { CryptoGlobalStats } from "../trading/coingecko";
import type { FearGreedSnapshot } from "../trading/fear-greed";
import type { TradingCommentary } from "./trading-commentary";

export interface TradingSection extends TradingCommentary {
  generated_at: string;
  tickers: TickerAnalysis[];
  crypto_fear_greed?: FearGreedSnapshot;
  crypto_global?: CryptoGlobalStats;
}

export interface ArticleInput extends RawArticle {
  source: string;
}

const PER_CATEGORY_LIMIT: Record<Category, number> = {
  tech: 25,
  finance: 20,
  politics: 15,
};

const MAX_AGE_DAYS = 14;

const ATTEMPT_CONFIGS = [
  { limits: PER_CATEGORY_LIMIT, excerptChars: 200 },
  {
    limits: {
      tech: 12,
      finance: 10,
      politics: 8,
    } satisfies Record<Category, number>,
    excerptChars: 100,
  },
  {
    limits: {
      tech: 6,
      finance: 6,
      politics: 4,
    } satisfies Record<Category, number>,
    excerptChars: 0,
  },
] as const;

/**
 * Pick `limit` items from `items` so every source gets a fair shot.
 *
 * Why this exists: the previous `slice(0, limit)` honored insertion order,
 * which is the source-iteration order in daily.ts. That gave whichever
 * source came first 100% of the quota — e.g. all 25 tech slots filled by
 * Hacker News before GitHub Trending / Solidot / V2EX / 阮一峰 got a turn.
 *
 * Strategy: drop items older than MAX_AGE_DAYS, group by sourceId,
 * sort each bucket newest-first, then round-robin one item per source
 * until we hit the limit. Sources with fewer items naturally drop out
 * and others absorb the slack.
 */
function selectRoundRobin(
  items: ArticleInput[],
  limit: number,
): ArticleInput[] {
  const cutoff = Date.now() - MAX_AGE_DAYS * 86_400_000;
  const fresh = items.filter(
    (it) => !it.publishedAt || it.publishedAt.getTime() >= cutoff,
  );

  const bySource = new Map<string, ArticleInput[]>();
  for (const it of fresh) {
    const arr = bySource.get(it.sourceId) ?? [];
    arr.push(it);
    bySource.set(it.sourceId, arr);
  }
  for (const arr of bySource.values()) {
    arr.sort(
      (a, b) =>
        (b.publishedAt?.getTime() ?? 0) - (a.publishedAt?.getTime() ?? 0),
    );
  }

  const buckets = Array.from(bySource.values());
  const out: ArticleInput[] = [];
  let madeProgress = true;
  while (out.length < limit && madeProgress) {
    madeProgress = false;
    for (const b of buckets) {
      if (b.length === 0) continue;
      out.push(b.shift()!);
      madeProgress = true;
      if (out.length >= limit) break;
    }
  }
  return out;
}

function parseBriefs(value: unknown, field: string): BriefItem[] {
  if (!Array.isArray(value)) {
    throw new LlmOutputError(
      "invalid_schema",
      `${field} is missing or not an array`,
    );
  }
  return value.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new LlmOutputError(
        "invalid_schema",
        `${field}[${index}] is not an object`,
      );
    }
    const candidate = item as Partial<BriefItem>;
    if (
      typeof candidate.title !== "string" ||
      !candidate.title.trim() ||
      typeof candidate.url !== "string" ||
      !candidate.url.trim() ||
      typeof candidate.source !== "string" ||
      !candidate.source.trim() ||
      typeof candidate.summary !== "string" ||
      !candidate.summary.trim() ||
      typeof candidate.importance !== "number" ||
      !Number.isFinite(candidate.importance) ||
      candidate.importance < 1 ||
      candidate.importance > 10
    ) {
      throw new LlmOutputError(
        "invalid_schema",
        `${field}[${index}] has an incomplete or invalid BriefItem shape`,
      );
    }
    return candidate as BriefItem;
  });
}

function requireText(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new LlmOutputError(
      "invalid_schema",
      `${field} is missing or empty`,
    );
  }
  return value.trim();
}

export function parseDailyReportText(
  text: string,
  requiredCategories: ReadonlySet<Category> = new Set(),
): DailyReport {
  const cleaned = extractJson(text);
  if (!cleaned) {
    throw new LlmOutputError("empty_output", "daily digest response was empty");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(cleaned);
  } catch (strictErr) {
    try {
      parsed = JSON.parse(jsonrepair(cleaned));
      console.warn("[pipeline] JSON.parse failed but jsonrepair recovered");
    } catch {
      const detail =
        strictErr instanceof Error ? strictErr.message : String(strictErr);
      throw new LlmOutputError(
        "invalid_json",
        `daily digest returned invalid JSON: ${detail}`,
      );
    }
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new LlmOutputError(
      "invalid_schema",
      "daily digest JSON root is not an object",
    );
  }
  const report = parsed as Record<string, unknown>;
  if (
    !Array.isArray(report.keywords) ||
    report.keywords.length === 0 ||
    report.keywords.some(
      (keyword) => typeof keyword !== "string" || !keyword.trim(),
    )
  ) {
    throw new LlmOutputError(
      "invalid_schema",
      "keywords is missing, empty, or invalid",
    );
  }

  const result: DailyReport = {
    hero_headline: requireText(report.hero_headline, "hero_headline"),
    daily_overview: requireText(report.daily_overview, "daily_overview"),
    tech_briefs: parseBriefs(report.tech_briefs, "tech_briefs"),
    finance_briefs: parseBriefs(report.finance_briefs, "finance_briefs"),
    politics_briefs: parseBriefs(report.politics_briefs, "politics_briefs"),
    editor_note: requireText(report.editor_note, "editor_note"),
    keywords: (report.keywords as string[]).map((keyword) => keyword.trim()),
  };
  const briefsByCategory: Record<Category, BriefItem[]> = {
    tech: result.tech_briefs,
    finance: result.finance_briefs,
    politics: result.politics_briefs,
  };
  for (const category of requiredCategories) {
    if (briefsByCategory[category].length === 0) {
      throw new LlmOutputError(
        "invalid_schema",
        `${category}_briefs is empty despite available candidates`,
      );
    }
  }
  return result;
}

async function dumpInvalidOutput(text: string, cleaned: string): Promise<void> {
  try {
    const fs = await import("node:fs");
    fs.mkdirSync("logs", { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    fs.writeFileSync(`logs/digest-raw-${ts}.txt`, text, "utf8");
    fs.writeFileSync(`logs/digest-cleaned-${ts}.txt`, cleaned, "utf8");
    console.warn(
      `[pipeline] invalid digest output saved to logs/digest-raw-${ts}.txt`,
    );
  } catch {
    // best-effort logging
  }
}

async function callOnce(
  userPayloadJson: string,
  runner: LlmRunner,
  requiredCategories: ReadonlySet<Category>,
): Promise<DailyReport> {
  // Claude Code CLI's built-in system prompt biases the model toward
  // conversational markdown output. Anchor the format expectation in the
  // user message (instruction recency wins) *and* explicitly demand every
  // schema field be populated — without this Sonnet has been observed to
  // emit a JSON shell with empty arrays to "satisfy" a JSON-only ask.
  const userPrompt =
    REPORT_LOCALE === "en"
      ? [
          "**Output language: ENGLISH ONLY.** Every string value in the JSON — hero_headline, daily_overview, every brief's title/summary, editor_note, keywords — must be written entirely in English. No Chinese characters anywhere.",
          "",
          "Your task: generate today's daily brief from the candidate news below. **The response MUST be a single valid JSON object** — starts with `{`, ends with `}`, no markdown, no code fences, no explanations.",
          "",
          "The JSON must contain every field non-empty (briefs arrays per the system-prompt counts):",
          "  - hero_headline: 10-25 word headline of the day",
          "  - daily_overview: **150-250 word** paragraph covering tech / finance / politics signals so a reader sees the whole picture at a glance",
          "  - tech_briefs: **3-5** tech BriefItems",
          "  - finance_briefs: **3-5** finance BriefItems",
          "  - politics_briefs: **2-3** politics BriefItems",
          "  - editor_note: 30-60 word editor's note",
          "  - keywords: 5-8 keywords",
          "",
          "BriefItem fields: title, url (copied verbatim from candidate), source, summary, importance (1-10).",
          "**Quote rule (important!)**: For any quotation INSIDE a JSON string, use single quotes ' or curly quotes '\" — **never** raw double quotes \", which break JSON parsing.",
          "No trailing commas.",
          "",
          `Candidate news (JSON array, ${userPayloadJson.length} chars):`,
          userPayloadJson,
        ].join("\n")
      : [
          "你的任务：根据下方候选新闻，生成一份当日简报，**响应必须是一个合法 JSON 对象**——以 `{` 开头，以 `}` 结尾，不要 markdown / 不要代码围栏 / 不要任何解释。",
          "",
          "JSON 必须包含全部字段且不能为空（briefs 数组按 system prompt 规定的条数填充）：",
          "  - hero_headline: 10-25 字的当日一句话头条",
          "  - daily_overview: **150-220 字** 的当日总览段落，一段话覆盖技术 / 财经 / 时政 的核心信号，让读者一眼抓住全貌",
          "  - tech_briefs: **3-5 条** 科技 BriefItem",
          "  - finance_briefs: **3-5 条** 财经 BriefItem",
          "  - politics_briefs: **2-3 条** 时政 BriefItem",
          "  - editor_note: 30-60 字的编辑短评",
          "  - keywords: 5-8 个关键词",
          "",
          "BriefItem 字段：title、url（必须从候选条目原样选取）、source、summary、importance(1-10)。",
          "**引号规则（重要！）**：JSON 字符串内的中文引用请使用**中文全角引号**「」或者 “”，**绝对不要**用英文双引号 \" —— 那会导致 JSON 解析失败。例：写 商务部回应「内卷」 而不是 商务部回应\"内卷\"。",
          "不要使用单引号、不要末尾多余逗号。",
          "",
          "候选新闻（JSON 数组，共 " + userPayloadJson.length + " 字符）：",
          userPayloadJson,
        ].join("\n");
  const { text } = await runner({
    systemPrompt: SYSTEM_PROMPT_DIGEST,
    userPrompt,
  });
  try {
    return parseDailyReportText(text, requiredCategories);
  } catch (error) {
    if (isLlmOutputError(error) && text.trim()) {
      await dumpInvalidOutput(text, extractJson(text));
    }
    throw error;
  }
}

function buildFallbackReport(articles: ArticleInput[]): DailyReport {
  const grouped: Record<Category, ArticleInput[]> = {
    tech: [],
    finance: [],
    politics: [],
  };
  for (const article of articles) grouped[article.category].push(article);

  const truncate = (value: string, maxChars: number): string =>
    value.length > maxChars ? `${value.slice(0, maxChars - 3)}...` : value;
  const toBrief = (article: ArticleInput): BriefItem => ({
    title: truncate(article.title.trim(), 120),
    url: article.url,
    source: article.source.trim(),
    summary: truncate(
      article.summary?.trim() ||
      article.excerpt?.trim().slice(0, 240) ||
      (REPORT_LOCALE === "en"
        ? "Open the source for the full report."
        : "详情请查看原文。"),
      240,
    ),
    importance: 5,
  });
  const tech = selectRoundRobin(grouped.tech, 3).map(toBrief);
  const finance = selectRoundRobin(grouped.finance, 3).map(toBrief);
  const politics = selectRoundRobin(grouped.politics, 2).map(toBrief);
  const lead = tech[0] ?? finance[0] ?? politics[0];
  const total = tech.length + finance.length + politics.length;

  if (REPORT_LOCALE === "en") {
    return {
      hero_headline: lead
        ? `Today's brief: ${truncate(lead.title, 80)}`
        : "Today's daily brief",
      daily_overview: `The editorial model did not return a complete digest, so this edition was assembled deterministically from ${total} recent items across the available technology, finance, and world-news sources. Links and source-provided summaries are preserved below.`,
      tech_briefs: tech,
      finance_briefs: finance,
      politics_briefs: politics,
      editor_note: "Automatic fallback edition; source links remain available for verification.",
      keywords: ["technology", "finance", "world", "daily brief", "sources"],
    };
  }
  return {
    hero_headline: lead
      ? `今日要闻：${truncate(lead.title, 40)}`
      : "今日资讯简报",
    daily_overview: `本期编辑模型未返回完整摘要，系统已从科技、财经与国际新闻来源中确定性整理 ${total} 条近期资讯。下方保留原始链接及已有来源摘要，便于继续阅读和核验。`,
    tech_briefs: tech,
    finance_briefs: finance,
    politics_briefs: politics,
    editor_note: "本期为自动降级版本，新闻来源与原文链接均予以保留。",
    keywords: ["科技", "财经", "国际", "每日简报", "多源资讯"],
  };
}

export async function generateDailyReport(
  articles: ArticleInput[],
  runner: LlmRunner = runLlm,
): Promise<{ report: DailyReport; tokensUsed: number }> {
  const grouped: Record<Category, ArticleInput[]> = {
    tech: [],
    finance: [],
    politics: [],
  };
  for (const a of articles) grouped[a.category].push(a);

  const fullCompact = (Object.keys(grouped) as Category[]).flatMap((category) =>
    selectRoundRobin(grouped[category], PER_CATEGORY_LIMIT[category]),
  );

  for (let attempt = 0; attempt < ATTEMPT_CONFIGS.length; attempt++) {
    const config = ATTEMPT_CONFIGS[attempt];
    const compact = (Object.keys(grouped) as Category[]).flatMap((category) =>
      selectRoundRobin(grouped[category], config.limits[category]),
    );
    const userPayload = compact.map((article, index) => ({
      n: index + 1,
      title: article.title,
      url: article.url,
      source: article.source,
      category: article.category,
      ...(config.excerptChars > 0
        ? { excerpt: (article.excerpt ?? "").slice(0, config.excerptChars) }
        : {}),
      published: article.publishedAt?.toISOString() ?? "",
    }));

    try {
      const requiredCategories = new Set(
        compact.map((article) => article.category),
      );
      const report = await callOnce(
        JSON.stringify(userPayload),
        runner,
        requiredCategories,
      );
      return { report, tokensUsed: 0 };
    } catch (error) {
      if (!isLlmOutputError(error)) throw error;
      const attemptNumber = attempt + 1;
      if (attemptNumber < ATTEMPT_CONFIGS.length) {
        console.warn(
          `[pipeline] digest attempt ${attemptNumber}/${ATTEMPT_CONFIGS.length} failed (${error.code}); retrying with fewer candidates: ${error.message}`,
        );
      } else {
        console.warn(
          `[pipeline] all ${ATTEMPT_CONFIGS.length} digest attempts returned unusable output; publishing deterministic fallback`,
        );
      }
    }
  }

  return { report: buildFallbackReport(fullCompact), tokensUsed: 0 };
}
