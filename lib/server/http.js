function sendJson(res, status, value) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(value));
}

async function readJson(req, maxBytes = 5 * 1024 * 1024) {
  if (req.body !== undefined) {
    let value = req.body;
    if (Buffer.isBuffer(value) || typeof value === "string") {
      try {
        value = JSON.parse(value.toString());
      } catch {
        const error = new Error("Request body must be valid JSON.");
        error.statusCode = 400;
        throw error;
      }
    }
    if (Buffer.byteLength(JSON.stringify(value)) > maxBytes) {
      const error = new Error("Request is too large.");
      error.statusCode = 413;
      throw error;
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      const error = new Error("Request body must be a JSON object.");
      error.statusCode = 400;
      throw error;
    }
    return value;
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error("Request is too large.");
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  let value;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    const error = new Error("Request body must be valid JSON.");
    error.statusCode = 400;
    throw error;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    const error = new Error("Request body must be a JSON object.");
    error.statusCode = 400;
    throw error;
  }
  return value;
}

function sameOrigin(req) {
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    const parsed = new URL(origin);
    const host = req.headers.host;
    return parsed.host === host && ["https:", "http:"].includes(parsed.protocol);
  } catch {
    return false;
  }
}

module.exports = { readJson, sameOrigin, sendJson };
