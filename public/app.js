const CONFIDENCE_THRESHOLD = 0.95;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const MAX_PROCESSED_IMAGE_BYTES = 2.8 * 1024 * 1024;
const MAX_IMAGE_SIDE = 1600;
const COMPLIANT_MASK_CLASSES = new Set(["with_mask"]);
const COMPLIANT_HELMET_CLASSES = new Set(["helmet"]);
const CONFIRMED_MASK_CLASSES = new Set(["without_mask", "mask_worn_incorrectly"]);
const CONFIRMED_HELMET_CLASSES = new Set(["no_helmet"]);

const byId = (id) => document.getElementById(id);
const imageInput = byId("image-input");
const dropZone = byId("drop-zone");
const analyzeButton = byId("analyze-button");
const resultCanvas = byId("result-canvas");
const resultContext = resultCanvas.getContext("2d");
const selectedFile = byId("selected-file");
let chosenFile = null;
let currentImage = null;
let previewUrl = null;

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}

function showAlert(message) {
  const alert = byId("app-alert");
  alert.textContent = message;
  alert.hidden = !message;
}

function switchPage(pageId) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.page === pageId;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-current", active ? "page" : "false");
  });
  document.querySelectorAll(".page").forEach((page) => { page.hidden = page.id !== pageId; });
  showAlert("");
  if (pageId === "evidence-page") loadEvents();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchPage(tab.dataset.page));
});

function setImage(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showAlert("Choose a JPG, PNG or WebP image.");
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showAlert("This image is larger than 12 MB. Choose a smaller photo.");
    return;
  }
  chosenFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile.replaceChildren();
  const name = document.createElement("span");
  name.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.addEventListener("click", clearImage);
  selectedFile.append(name, remove);
  selectedFile.hidden = false;
  analyzeButton.disabled = false;
  showAlert("");

  const preview = new Image();
  preview.onload = () => {
    currentImage = preview;
    drawImage([]);
    byId("result-empty").hidden = true;
    byId("result-content").hidden = false;
    byId("result-badge").textContent = "READY TO ANALYZE";
    byId("result-badge").className = "result-badge ready";
    byId("result-note").textContent = "Image selected. Run the analysis to see detections.";
    byId("prediction-list").replaceChildren();
  };
  preview.onerror = () => showAlert("The selected image could not be opened.");
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
}

function clearImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  chosenFile = null;
  currentImage = null;
  imageInput.value = "";
  selectedFile.hidden = true;
  analyzeButton.disabled = true;
  byId("result-content").hidden = true;
  byId("result-empty").hidden = false;
  byId("result-badge").textContent = "AWAITING PHOTO";
  byId("result-badge").className = "result-badge";
}

imageInput.addEventListener("change", () => setImage(imageInput.files[0]));
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  setImage(event.dataTransfer.files[0]);
});

async function imageAsDataUrl(image) {
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  for (const quality of [0.86, 0.76, 0.66, 0.56]) {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    if (!blob) throw new Error("The selected image could not be prepared for analysis.");
    if (blob.size <= MAX_PROCESSED_IMAGE_BYTES) {
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("The selected image could not be prepared for analysis."));
        reader.readAsDataURL(blob);
      });
    }
  }
  throw new Error("The resized image is still too large. Choose a smaller photo.");
}

function isViolation(task, label) {
  return task === "mask" ? CONFIRMED_MASK_CLASSES.has(label) : CONFIRMED_HELMET_CLASSES.has(label);
}

function isCompliant(task, label) {
  return task === "mask" ? COMPLIANT_MASK_CLASSES.has(label) : COMPLIANT_HELMET_CLASSES.has(label);
}

function drawImage(predictions) {
  if (!currentImage) return;
  resultCanvas.width = currentImage.naturalWidth;
  resultCanvas.height = currentImage.naturalHeight;
  resultContext.drawImage(currentImage, 0, 0);
  const task = byId("task-select").value;
  predictions.forEach((prediction) => {
    const [x, y, width, height] = prediction.box;
    const left = x * resultCanvas.width;
    const top = y * resultCanvas.height;
    const boxWidth = width * resultCanvas.width;
    const boxHeight = height * resultCanvas.height;
    const violation = isViolation(task, prediction.label);
    const confirmed = prediction.confidence >= CONFIDENCE_THRESHOLD;
    const color = !confirmed || (!violation && !isCompliant(task, prediction.label))
      ? "#ffc36b"
      : violation ? "#ff776e" : "#62e0ad";
    resultContext.strokeStyle = color;
    resultContext.lineWidth = Math.max(2, resultCanvas.width / 400);
    if (!confirmed) resultContext.setLineDash([resultContext.lineWidth * 3, resultContext.lineWidth * 2]);
    resultContext.strokeRect(left, top, boxWidth, boxHeight);
    resultContext.setLineDash([]);
    const title = `${prediction.label.replaceAll("_", " ")} · ${Math.round(prediction.confidence * 100)}%`;
    resultContext.font = `600 ${Math.max(12, resultCanvas.width / 70)}px sans-serif`;
    const textWidth = resultContext.measureText(title).width;
    const labelHeight = Math.max(23, resultCanvas.width / 36);
    resultContext.fillStyle = color;
    resultContext.fillRect(left, Math.max(0, top - labelHeight), textWidth + 12, labelHeight);
    resultContext.fillStyle = "#07111b";
    resultContext.fillText(title, left + 6, Math.max(15, top - labelHeight / 2 + 5));
  });
}

function cropAsDataUrl(prediction) {
  const [x, y, width, height] = prediction.box;
  const sourceWidth = currentImage.naturalWidth;
  const sourceHeight = currentImage.naturalHeight;
  const left = Math.max(0, Math.floor(x * sourceWidth));
  const top = Math.max(0, Math.floor(y * sourceHeight));
  const right = Math.min(sourceWidth, Math.ceil((x + width) * sourceWidth));
  const bottom = Math.min(sourceHeight, Math.ceil((y + height) * sourceHeight));
  if (right <= left || bottom <= top) return null;
  const crop = document.createElement("canvas");
  const factor = Math.min(1, 512 / Math.max(right - left, bottom - top));
  crop.width = Math.max(1, Math.round((right - left) * factor));
  crop.height = Math.max(1, Math.round((bottom - top) * factor));
  crop.getContext("2d").drawImage(currentImage, left, top, right - left, bottom - top, 0, 0, crop.width, crop.height);
  return crop.toDataURL("image/jpeg", 0.88);
}

function renderPredictions(predictions) {
  const task = byId("task-select").value;
  const list = byId("prediction-list");
  list.replaceChildren();
  predictions.forEach((prediction) => {
    const row = document.createElement("div");
    row.className = "prediction-item";
    const name = document.createElement("span");
    name.className = "prediction-name";
    const mark = document.createElement("span");
    const violation = isViolation(task, prediction.label);
    mark.className = `prediction-mark ${prediction.confidence < CONFIDENCE_THRESHOLD ? "review" : violation ? "violation" : ""}`;
    const label = document.createElement("span");
    label.textContent = prediction.label.replaceAll("_", " ");
    name.append(mark, label);
    const score = document.createElement("strong");
    score.textContent = `${Math.round(prediction.confidence * 100)}%`;
    row.append(name, score);
    list.append(row);
  });
}

analyzeButton.addEventListener("click", async () => {
  if (!chosenFile || !currentImage) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";
  byId("result-badge").textContent = "ANALYZING";
  byId("result-badge").className = "result-badge";
  showAlert("");
  try {
    const task = byId("task-select").value;
    const result = await apiRequest("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ task, image: await imageAsDataUrl(currentImage) }),
    });
    const predictions = result.predictions;
    const confirmed = predictions.filter((prediction) => prediction.confidence >= CONFIDENCE_THRESHOLD);
    const violations = confirmed.filter((prediction) => isViolation(task, prediction.label));
    const compliant = confirmed.filter((prediction) => isCompliant(task, prediction.label)).length;
    drawImage(predictions);
    renderPredictions(predictions);
    byId("metric-total").textContent = predictions.length;
    byId("metric-violations").textContent = violations.length;
    byId("metric-compliant").textContent = compliant;
    byId("result-badge").textContent = "ANALYSIS COMPLETE";
    byId("result-badge").className = "result-badge ready";
    byId("result-note").textContent = predictions.length === 0
      ? "No regions were returned. This is inconclusive, not confirmation of compliance."
      : `${confirmed.length} confirmed region${confirmed.length === 1 ? "" : "s"} at the fixed 95% threshold. Lower-confidence results are review suggestions.`;

    if (byId("save-evidence").checked && violations.length) {
      const outcomes = await Promise.allSettled(violations.map((prediction) => {
        const evidence = cropAsDataUrl(prediction);
        if (!evidence) throw new Error("A violation crop could not be prepared.");
        return apiRequest("/api/events", {
          method: "POST",
          body: JSON.stringify({
            task,
            class_name: prediction.label,
            confidence: prediction.confidence,
            source: "image upload",
            evidence,
          }),
        });
      }));
      const failed = outcomes.filter((outcome) => outcome.status === "rejected").length;
      if (failed) showAlert(`${violations.length - failed} crop(s) saved. ${failed} could not be saved; check the Supabase configuration and try again.`);
      await loadEvents();
    }
  } catch (error) {
    byId("result-badge").textContent = "ANALYSIS FAILED";
    byId("result-badge").className = "result-badge failed";
    showAlert(error.message);
  } finally {
    analyzeButton.disabled = !chosenFile;
    analyzeButton.innerHTML = 'Analyze photo <span aria-hidden="true">↗</span>';
  }
});

function renderEvents(events) {
  const grid = byId("events-grid");
  grid.replaceChildren();
  byId("events-empty").hidden = events.length > 0;
  byId("event-count").textContent = String(events.length);
  events.forEach((event) => {
    const card = document.createElement("article");
    card.className = "event-card";
    const image = document.createElement("img");
    image.src = event.evidence_url;
    image.alt = `${event.task} violation evidence crop`;
    image.loading = "lazy";
    const info = document.createElement("div");
    info.className = "event-info";
    const title = document.createElement("div");
    title.className = "event-title";
    const label = document.createElement("span");
    label.textContent = String(event.class_name).replaceAll("_", " ");
    const confidence = document.createElement("span");
    confidence.className = "event-confidence";
    confidence.textContent = `${Math.round(event.confidence * 100)}%`;
    title.append(label, confidence);
    const meta = document.createElement("p");
    meta.className = "event-meta";
    meta.textContent = `${event.task === "mask" ? "Face mask" : "Safety helmet"} · ${new Date(event.timestamp_utc).toLocaleString()} · ${event.source}`;
    const remove = document.createElement("button");
    remove.className = "button";
    remove.type = "button";
    remove.textContent = "Delete crop and record";
    remove.addEventListener("click", () => deleteEvent(event.id));
    info.append(title, meta, remove);
    card.append(image, info);
    grid.append(card);
  });
}

async function loadEvents() {
  const filter = byId("evidence-filter").value;
  const query = filter ? `?task=${encodeURIComponent(filter)}` : "";
  try {
    const result = await apiRequest(`/api/events${query}`);
    renderEvents(result.events);
  } catch (error) {
    showAlert(error.message);
  }
}

async function deleteEvent(id) {
  try {
    await apiRequest(`/api/events?id=${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadEvents();
  } catch (error) {
    showAlert(error.message);
  }
}

byId("evidence-filter").addEventListener("change", loadEvents);
byId("refresh-events").addEventListener("click", loadEvents);
loadEvents();
