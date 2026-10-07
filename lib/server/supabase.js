function supabaseConfig() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const bucket = process.env.SUPABASE_EVIDENCE_BUCKET || "safety-evidence";
  if (!url || !key) throw new Error("Supabase storage is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Vercel.");
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("SUPABASE_URL must be a valid HTTPS URL.");
  }
  if (parsed.protocol !== "https:") throw new Error("SUPABASE_URL must use HTTPS.");
  return { url: parsed.origin, key, bucket };
}

async function supabaseFetch(path, options = {}) {
  const config = supabaseConfig();
  const response = await fetch(`${config.url}${path}`, {
    ...options,
    headers: {
      apikey: config.key,
      Authorization: `Bearer ${config.key}`,
      ...options.headers,
    },
  });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Supabase request failed (${response.status}): ${body.slice(0, 500)}`);
  }
  return body ? JSON.parse(body) : null;
}

function storagePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

async function uploadEvidence(path, bytes) {
  const { bucket } = supabaseConfig();
  return supabaseFetch(`/storage/v1/object/${encodeURIComponent(bucket)}/${storagePath(path)}`, {
    method: "POST",
    headers: { "Content-Type": "image/jpeg", "x-upsert": "false" },
    body: bytes,
  });
}

async function removeEvidence(paths) {
  if (!paths.length) return;
  const { bucket } = supabaseConfig();
  return supabaseFetch(`/storage/v1/object/${encodeURIComponent(bucket)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prefixes: paths }),
  });
}

async function signedEvidenceUrl(path) {
  const { bucket, url } = supabaseConfig();
  const result = await supabaseFetch(`/storage/v1/object/sign/${encodeURIComponent(bucket)}/${storagePath(path)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expiresIn: 3600 }),
  });
  if (!result?.signedURL) throw new Error("Supabase did not return a signed evidence URL.");
  return result.signedURL.startsWith("http") ? result.signedURL : `${url}/storage/v1${result.signedURL}`;
}

module.exports = { removeEvidence, signedEvidenceUrl, supabaseFetch, supabaseConfig, uploadEvidence };
