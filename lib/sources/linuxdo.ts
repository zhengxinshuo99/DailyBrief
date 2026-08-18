import Parser from "rss-parser";
import { curlFetch } from "./curl-fetch";
import type { RawArticle } from "./types";
import { V2EX_OFF_TOPIC_RE } from "./v2ex";

const HEADERS: Record<string, string> = {
  "User-Agent":
    "Mozilla/5.0 (compatible; DailyBriefBot/1.0; +https://github.com/leiting-eric/DailyBrief)",
  Accept:
    "application/atom+xml, application/rss+xml, application/xml, text/xml, */*",
  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
};

const parser = new Parser({ timeout: 15000 });
const DEFAULT_PRIMARY_URL = "https://linux.do/top.rss?period=daily";
const DEFAULT_FALLBACK_URL = "https://linux.do/latest.rss";

type FetchText = (
  url: string,
  headers?: Record<string, string>,
) => Promise<string>;

export interface LinuxDoFetchOptions {
  limit?: number;
  primaryUrl?: string;
  fallbackUrl?: string;
  fetchText?: FetchText;
}

function stripHtml(s: string): string {
  return s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

function isCloudflareChallenge(text: string): boolean {
  const head = text.slice(0, 500).toLowerCase();
  return (
    head.includes("just a moment") ||
    head.includes("cf-chl") ||
    (head.startsWith("<!doctype html") && head.includes("cloudflare"))
  );
}

function endpointLabel(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname}${parsed.pathname}`;
  } catch {
    return "invalid-url";
  }
}

function errorDetail(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const curlError = message.match(/curl: \(\d+\)[^\r\n]*/)?.[0];
  return curlError ?? message.replace(/\s+/g, " ").trim().slice(0, 240);
}

async function fetchFeed(url: string, fetchText: FetchText) {
  const xml = await fetchText(url, HEADERS);
  if (!xml.trim()) {
    throw new Error("empty response");
  }
  if (isCloudflareChallenge(xml)) {
    throw new Error("cloudflare challenge page");
  }
  const head = xml.trimStart().slice(0, 1000).toLowerCase();
  if (head.startsWith("<!doctype html") || head.startsWith("<html")) {
    throw new Error(`HTML response (${xml.length} chars)`);
  }
  if (!head.includes("<rss") && !head.includes("<feed")) {
    throw new Error(`non-RSS response (${xml.length} chars)`);
  }
  try {
    return await parser.parseString(xml);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid RSS XML (${xml.length} chars): ${message}`);
  }
}

/**
 * LinuxDo data source.
 *
 * Uses LinuxDo's public Discourse RSS feeds — the same URLs that any RSS
 * reader subscribes to. RSS is the syndication protocol the site exposes
 * to third-party aggregators, so this fetcher identifies itself honestly
 * as `DailyBriefBot/1.0` (no UA spoofing).
 *
 * Strategy: try /top.rss?period=daily first (matches "today's hot"
 * semantics), fall back to /latest.rss when /top fails.
 *
 * Cloudflare still TLS-fingerprints Node's undici on linux.do, so we
 * shell out to curl (see lib/sources/curl-fetch.ts). RSS endpoints sit
 * on Cloudflare's syndication-friendly path and rarely trigger the
 * "Just a moment…" interstitial.
 */
export async function fetchLinuxDo(
  sourceId: string,
  options: LinuxDoFetchOptions = {},
): Promise<RawArticle[]> {
  const {
    limit = 25,
    primaryUrl = DEFAULT_PRIMARY_URL,
    fallbackUrl = DEFAULT_FALLBACK_URL,
    fetchText = curlFetch,
  } = options;
  const failures: string[] = [];
  let feed;
  try {
    feed = await fetchFeed(primaryUrl, fetchText);
  } catch (error) {
    failures.push(`primary ${endpointLabel(primaryUrl)}: ${errorDetail(error)}`);
    console.warn(`[linuxdo] ${failures[0]}; trying fallback`);
    try {
      feed = await fetchFeed(fallbackUrl, fetchText);
    } catch (fallbackError) {
      failures.push(
        `fallback ${endpointLabel(fallbackUrl)}: ${errorDetail(fallbackError)}`,
      );
      throw new Error(`LinuxDo RSS unavailable — ${failures.join("; ")}`);
    }
  }

  return (feed.items ?? [])
    .filter(
      (item) =>
        item.title && item.link && !V2EX_OFF_TOPIC_RE.test(item.title),
    )
    .slice(0, limit)
    .map((item) => ({
      sourceId,
      title: (item.title ?? "").trim(),
      url: (item.link ?? "").trim(),
      excerpt: stripHtml(item.contentSnippet ?? item.content ?? "").slice(
        0,
        300,
      ),
      publishedAt: item.isoDate ? new Date(item.isoDate) : undefined,
      category: "tech" as const,
    }));
}
