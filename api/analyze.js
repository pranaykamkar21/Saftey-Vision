const { readJson, sameOrigin, sendJson } = require("../lib/server/http");

const MAX_REQUEST_BYTES = 5 * 1024 * 1024;
const MAX_IMAGE_BYTES = 3 * 1024 * 1024;
const VALID_TASKS = new Set(["mask", "helmet"]);
const VALID_LABELS = {
  mask: new Set(["with_mask", "without_mask", "mask_worn_incorrectly"]),
  helmet: new Set(["helmet", "no_helmet"]),
};

function validPrediction(prediction, task) {
  if (!prediction || typeof prediction.label !== "string" || !VALID_LABELS[task].has(prediction.label)) return false;
  if (!Number.isFinite(prediction.confidence) || prediction.confidence < 0 || prediction.confidence > 1) return false;
  if (!Array.isArray(prediction.box) || prediction.box.length !== 4 || !prediction.box.every(Number.isFinite)) return false;
  const [x, y, width, height] = prediction.box;
  return x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.001 && y + height <= 1.001;
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { error: "Method not allowed." });
  }
  if (!sameOrigin(req)) return sendJson(res, 403, { error: "Cross-origin requests are not allowed." });
  const endpoint = process.env.INFERENCE_API_URL;
  if (!endpoint) return sendJson(res, 503, { error: "Inference is not configured. Set INFERENCE_API_URL in Vercel." });

  let parsedEndpoint;
  try {
    parsedEndpoint = new URL(endpoint);
  } catch {
    return sendJson(res, 500, { error: "INFERENCE_API_URL must be a valid HTTPS URL." });
  }
  if (parsedEndpoint.protocol !== "https:") return sendJson(res, 500, { error: "INFERENCE_API_URL must use HTTPS." });

  try {
    const body = await readJson(req, MAX_REQUEST_BYTES);
    if (!VALID_TASKS.has(body.task)) return sendJson(res, 400, { error: "task must be mask or helmet." });
    const threshold = body.threshold ?? 0.95;
    if (!Number.isFinite(threshold) || threshold < 0.25 || threshold > 1) {
      return sendJson(res, 400, { error: "threshold must be a number between 0.25 and 1." });
    }
    if (typeof body.image !== "string" || !/^data:image\/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+=*$/.test(body.image)) {
      return sendJson(res, 400, { error: "image must be a base64 JPEG, PNG or WebP data URL." });
    }
    const encodedImage = body.image.slice(body.image.indexOf(",") + 1);
    if (Buffer.byteLength(encodedImage, "base64") > MAX_IMAGE_BYTES) {
      return sendJson(res, 413, { error: "Processed image exceeds the 3 MB limit." });
    }
    const headers = { "Content-Type": "application/json", Accept: "application/json" };
    if (process.env.INFERENCE_API_KEY) headers.Authorization = `Bearer ${process.env.INFERENCE_API_KEY}`;
    const response = await fetch(parsedEndpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ task: body.task, image: body.image, threshold }),
      signal: AbortSignal.timeout(50000),
    });
    if (!response.ok) {
      return sendJson(res, 502, { error: `Inference service returned an error (${response.status}).` });
    }
    let output;
    try {
      output = await response.json();
    } catch {
      return sendJson(res, 502, { error: "Inference service returned invalid JSON." });
    }
    if (!output || !Array.isArray(output.predictions) || output.predictions.length > 500 || !output.predictions.every((prediction) => validPrediction(prediction, body.task))) {
      return sendJson(res, 502, { error: "Inference response must contain predictions with label, confidence, and normalized [x, y, width, height] boxes." });
    }
    return sendJson(res, 200, { predictions: output.predictions });
  } catch (error) {
    if (error.statusCode) return sendJson(res, error.statusCode, { error: error.message });
    if (error.name === "TimeoutError" || error.name === "AbortError") {
      return sendJson(res, 504, { error: "Inference service timed out. Try again or check its availability." });
    }
    return sendJson(res, 502, { error: `Could not reach the inference service: ${error.message}` });
  }
};
