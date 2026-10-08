const { randomUUID } = require("node:crypto");
const { readJson, sameOrigin, sendJson } = require("../lib/server/http");
const { removeEvidence, signedEvidenceUrl, supabaseFetch, supabaseConfig, uploadEvidence } = require("../lib/server/supabase");

const RETENTION_DAYS = 30;
const ALLOWED_CLASSES = new Set(["without_mask", "mask_worn_incorrectly", "no_helmet"]);
const EVIDENCE_LIMIT = 1024 * 1024;

function tableUrl(query) {
  return `/rest/v1/violations?${query}`;
}

async function purgeExpired() {
  const cutoff = new Date(Date.now() - RETENTION_DAYS * 24 * 60 * 60 * 1000).toISOString();
  const expired = await supabaseFetch(tableUrl(`select=id,evidence_path&timestamp_utc=lt.${encodeURIComponent(cutoff)}&limit=100`));
  if (!expired.length) return;
  await removeEvidence(expired.map((row) => row.evidence_path));
  const ids = expired.map((row) => row.id);
  await supabaseFetch(tableUrl(`id=in.(${ids.join(",")})`), { method: "DELETE" });
}

async function listEvents(req, res) {
  const task = req.query?.task;
  if (task && !["mask", "helmet"].includes(task)) return sendJson(res, 400, { error: "task filter must be mask or helmet." });
  await purgeExpired();
  const cutoff = new Date(Date.now() - RETENTION_DAYS * 24 * 60 * 60 * 1000).toISOString();
  const filters = [`select=id,timestamp_utc,task,class_name,confidence,evidence_path,source,evidence_kind&timestamp_utc=gte.${encodeURIComponent(cutoff)}`];
  if (task) filters.push(`task=eq.${encodeURIComponent(task)}`);
  filters.push("order=timestamp_utc.desc", "limit=100");
  const rows = await supabaseFetch(tableUrl(filters.join("&")));
  const events = await Promise.all(rows.map(async (row) => ({
    ...row,
    evidence_url: await signedEvidenceUrl(row.evidence_path),
    evidence_path: undefined,
  })));
  return sendJson(res, 200, { events });
}

async function createEvent(req, res) {
  const body = await readJson(req, 2 * 1024 * 1024);
  if (!["mask", "helmet"].includes(body.task)) return sendJson(res, 400, { error: "task must be mask or helmet." });
  if (!ALLOWED_CLASSES.has(body.class_name)) return sendJson(res, 400, { error: "Unsupported violation class." });
  if ((body.task === "mask") !== (body.class_name !== "no_helmet")) {
    return sendJson(res, 400, { error: "The violation class does not match the selected profile." });
  }
  const threshold = body.threshold ?? 0.95;
  if (!Number.isFinite(threshold) || threshold < 0.25 || threshold > 1) {
    return sendJson(res, 400, { error: "threshold must be a number between 0.25 and 1." });
  }
  if (!Number.isFinite(body.confidence) || body.confidence < threshold || body.confidence > 1) {
    return sendJson(res, 400, { error: `Only violations at or above ${Math.round(threshold * 100)}% confidence can be saved.` });
  }
  if (typeof body.source !== "string" || body.source.length > 200) {
    return sendJson(res, 400, { error: "source must be a string of at most 200 characters." });
  }
  const match = typeof body.evidence === "string" && body.evidence.match(/^data:image\/jpeg;base64,([A-Za-z0-9+/]+=*)$/);
  if (!match) return sendJson(res, 400, { error: "evidence must be a base64 JPEG crop." });
  const bytes = Buffer.from(match[1], "base64");
  if (!bytes.length || bytes.length > EVIDENCE_LIMIT) return sendJson(res, 413, { error: "Evidence crop must be smaller than 1 MB." });

  supabaseConfig();
  const now = new Date();
  const path = `${now.getUTCFullYear()}/${String(now.getUTCMonth() + 1).padStart(2, "0")}/${randomUUID()}.jpg`;
  await uploadEvidence(path, bytes);
  const row = {
    timestamp_utc: now.toISOString(),
    task: body.task,
    class_name: body.class_name,
    confidence: body.confidence,
    evidence_path: path,
    source: body.source,
    evidence_kind: body.task === "mask" ? "face_crop" : "head_crop",
  };
  try {
    const inserted = await supabaseFetch(tableUrl("select=id,timestamp_utc,task,class_name,confidence,evidence_path,source,evidence_kind"), {
      method: "POST",
      headers: { "Content-Type": "application/json", Prefer: "return=representation" },
      body: JSON.stringify(row),
    });
    return sendJson(res, 201, { event: inserted[0] });
  } catch (error) {
    await removeEvidence([path]);
    throw error;
  }
}

async function deleteEvent(req, res) {
  const id = req.query?.id;
  if (typeof id !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) {
    return sendJson(res, 400, { error: "A valid event id is required." });
  }
  const rows = await supabaseFetch(tableUrl(`select=id,evidence_path&id=eq.${encodeURIComponent(id)}&limit=1`));
  if (!rows.length) return sendJson(res, 404, { error: "Evidence record not found." });
  await supabaseFetch(tableUrl(`id=eq.${encodeURIComponent(id)}`), { method: "DELETE" });
  await removeEvidence([rows[0].evidence_path]);
  return sendJson(res, 200, { ok: true });
}

module.exports = async function handler(req, res) {
  if (!["GET", "POST", "DELETE"].includes(req.method)) {
    res.setHeader("Allow", "GET, POST, DELETE");
    return sendJson(res, 405, { error: "Method not allowed." });
  }
  if (req.method !== "GET" && !sameOrigin(req)) return sendJson(res, 403, { error: "Cross-origin requests are not allowed." });
  try {
    if (req.method === "GET") return await listEvents(req, res);
    if (req.method === "POST") return await createEvent(req, res);
    return await deleteEvent(req, res);
  } catch (error) {
    return sendJson(res, 502, { error: error.message || "Evidence storage request failed." });
  }
};
