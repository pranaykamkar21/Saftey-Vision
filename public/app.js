const CONFIDENCE_THRESHOLD = 0.95;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const MAX_VIDEO_BYTES = 50 * 1024 * 1024;
const MAX_PROCESSED_IMAGE_BYTES = 2.8 * 1024 * 1024;
const MAX_IMAGE_SIDE = 1600;
const FRAME_INTERVAL_MS = 1500;
const TRACK_RETENTION_SECONDS = 8;
const COMPLIANT_MASK_CLASSES = new Set(["with_mask"]);
const COMPLIANT_HELMET_CLASSES = new Set(["helmet"]);
const CONFIRMED_MASK_CLASSES = new Set(["without_mask", "mask_worn_incorrectly"]);
const CONFIRMED_HELMET_CLASSES = new Set(["no_helmet"]);

const byId = (id) => document.getElementById(id);
const imageInput = byId("image-input");
const dropZone = byId("drop-zone");
const analyzeButton = byId("analyze-button");
const videoInput = byId("video-input");
const resultCanvas = byId("result-canvas");
const resultContext = resultCanvas.getContext("2d");
const selectedFile = byId("selected-file");
const selectedVideo = byId("selected-video");
const sourceVideo = byId("source-video");
const cameraPreview = byId("camera-preview");
const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d");
let chosenFile = null;
let chosenVideoFile = null;
let currentImage = null;
let previewUrl = null;
let videoUrl = null;
let sourceMode = "image";
let cameraStream = null;
let activeVideo = null;
let animationFrame = 0;
let analysisPending = false;
let processingStopped = false;
let mediaRecorder = null;
let recordingChunks = [];
let recordingType = "";
let lastAnalysisAt = 0;
let processedFrames = 0;
let totalDetections = 0;
let totalViolations = 0;
let totalCompliant = 0;
let activeViolationTracks = [];
let nextTrackId = 1;
let latestPredictions = [];
let runMode = null;
let runSourceLabel = "";
let downloadUrl = null;
let pendingEvidenceSaves = [];
let evidenceSaveErrors = 0;
let runToken = 0;
let finishingRun = false;

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
  if (pageId !== "analysis-page") stopActiveSources();
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

function setSourceMode(mode) {
  if (sourceMode === mode) return;
  stopActiveSources();
  sourceMode = mode;
  document.querySelectorAll(".source-mode").forEach((button) => {
    const active = button.dataset.source === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  byId("image-picker").hidden = mode !== "image";
  byId("video-picker").hidden = mode !== "video";
  byId("webcam-controls").hidden = mode !== "webcam";
  analyzeButton.hidden = mode !== "image";
  byId("video-start").hidden = mode !== "video";
  if (mode !== "image") {
    clearImage();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  if (mode !== "video") clearVideo();
  byId("video-download").hidden = true;
  byId("result-empty").hidden = false;
  byId("result-content").hidden = true;
  byId("result-badge").textContent = mode === "webcam" ? "CAMERA READY" : "AWAITING INPUT";
  byId("result-badge").className = "result-badge";
  byId("process-progress").hidden = true;
  byId("camera-start").hidden = false;
  byId("camera-stop").hidden = true;
  cameraPreview.hidden = true;
  cameraPreview.srcObject = null;
  byId("camera-preview-empty").hidden = false;
  byId("camera-preview-status").textContent = "CAMERA OFF";
  byId("camera-preview-frame").classList.remove("live");
  byId("task-select").disabled = false;
  showAlert("");
}

document.querySelectorAll(".source-mode").forEach((button) => {
  button.addEventListener("click", () => setSourceMode(button.dataset.source));
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

function setVideo(file) {
  if (!file) return;
  if (!["video/mp4", "video/webm", "video/quicktime"].includes(file.type)) {
    showAlert("Choose an MP4, WebM, or MOV video. Browser support may vary by format.");
    return;
  }
  if (file.size > MAX_VIDEO_BYTES) {
    showAlert("This video is larger than 50 MB. Choose a smaller clip.");
    return;
  }
  clearVideo();
  chosenVideoFile = file;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  sourceVideo.src = videoUrl;
  sourceVideo.onloadedmetadata = () => {
    if (!Number.isFinite(sourceVideo.duration) || sourceVideo.duration <= 0) {
      showAlert("This video has no readable duration. Try another MP4 or WebM clip.");
      byId("video-start").disabled = true;
      return;
    }
    if (sourceVideo.duration > 120) {
      showAlert("Video clips must be 2 minutes or shorter.");
      byId("video-start").disabled = true;
      return;
    }
    byId("video-start").disabled = false;
    showAlert("");
  };
  sourceVideo.onerror = () => {
    showAlert("Your browser could not open this video. Try an MP4 or WebM clip.");
    byId("video-start").disabled = true;
  };
  sourceVideo.load();
  const label = document.createElement("span");
  label.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.addEventListener("click", clearVideo);
  selectedVideo.replaceChildren(label, remove);
  selectedVideo.hidden = false;
  byId("video-start").disabled = false;
  showAlert("");
}

function clearVideo() {
  chosenVideoFile = null;
  videoInput.value = "";
  selectedVideo.hidden = true;
  byId("video-start").disabled = true;
  sourceVideo.onloadedmetadata = null;
  sourceVideo.onerror = null;
  sourceVideo.pause();
  sourceVideo.removeAttribute("src");
  sourceVideo.load();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
}

imageInput.addEventListener("change", () => setImage(imageInput.files[0]));
videoInput.addEventListener("change", () => setVideo(videoInput.files[0]));
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  setImage(event.dataTransfer.files[0]);
});
const videoDropZone = byId("video-drop-zone");
videoDropZone.addEventListener("dragover", (event) => { event.preventDefault(); videoDropZone.classList.add("dragging"); });
videoDropZone.addEventListener("dragleave", () => videoDropZone.classList.remove("dragging"));
videoDropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  videoDropZone.classList.remove("dragging");
  setVideo(event.dataTransfer.files[0]);
});

function sourceDimensions(source) {
  return {
    width: source.videoWidth || source.naturalWidth || source.width,
    height: source.videoHeight || source.naturalHeight || source.height,
  };
}

function prepareFrame(source) {
  const { width, height } = sourceDimensions(source);
  if (!width || !height) throw new Error("The video frame is not ready yet.");
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(width, height));
  frameCanvas.width = Math.max(1, Math.round(width * scale));
  frameCanvas.height = Math.max(1, Math.round(height * scale));
  frameContext.drawImage(source, 0, 0, frameCanvas.width, frameCanvas.height);
  return frameCanvas;
}

async function imageAsDataUrl(source) {
  const { width, height } = sourceDimensions(source);
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
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

function drawImage(predictions, source = currentImage) {
  if (!source) return;
  const { width, height } = sourceDimensions(source);
  if (!width || !height) return;
  const displayScale = Math.min(1, 1600 / Math.max(width, height));
  const displayWidth = Math.max(1, Math.round(width * displayScale));
  const displayHeight = Math.max(1, Math.round(height * displayScale));
  if (resultCanvas.width !== displayWidth || resultCanvas.height !== displayHeight) {
    resultCanvas.width = displayWidth;
    resultCanvas.height = displayHeight;
  }
  resultContext.drawImage(source, 0, 0, displayWidth, displayHeight);
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

function cropAsDataUrl(prediction, source = currentImage) {
  if (!source) return null;
  const [x, y, width, height] = prediction.box;
  const { width: sourceWidth, height: sourceHeight } = sourceDimensions(source);
  const left = Math.max(0, Math.floor(x * sourceWidth));
  const top = Math.max(0, Math.floor(y * sourceHeight));
  const right = Math.min(sourceWidth, Math.ceil((x + width) * sourceWidth));
  const bottom = Math.min(sourceHeight, Math.ceil((y + height) * sourceHeight));
  if (right <= left || bottom <= top) return null;
  const crop = document.createElement("canvas");
  const factor = Math.min(1, 512 / Math.max(right - left, bottom - top));
  crop.width = Math.max(1, Math.round((right - left) * factor));
  crop.height = Math.max(1, Math.round((bottom - top) * factor));
  crop.getContext("2d").drawImage(source, left, top, right - left, bottom - top, 0, 0, crop.width, crop.height);
  return crop.toDataURL("image/jpeg", 0.82);
}

async function requestPredictions(source, task) {
  const image = await imageAsDataUrl(source);
  const result = await apiRequest("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ task, image }),
  });
  return result.predictions;
}

function boxOverlap(first, second) {
  const [ax, ay, aw, ah] = first.box;
  const [bx, by, bw, bh] = second.box;
  const left = Math.max(ax, bx);
  const top = Math.max(ay, by);
  const right = Math.min(ax + aw, bx + bw);
  const bottom = Math.min(ay + ah, by + bh);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = aw * ah + bw * bh - intersection;
  return union > 0 ? intersection / union : 0;
}

function trackAndSaveViolations(predictions, frame, timestampSeconds) {
  activeViolationTracks = activeViolationTracks.filter(
    (track) => timestampSeconds - track.lastSeen <= TRACK_RETENTION_SECONDS,
  );
  const seenTrackIds = new Set();
  const task = byId("task-select").value;
  for (const prediction of predictions) {
    if (prediction.confidence < CONFIDENCE_THRESHOLD || !isViolation(task, prediction.label)) continue;
    const match = activeViolationTracks
      .filter((track) => track.label === prediction.label && !seenTrackIds.has(track.id))
      .map((track) => ({ track, overlap: boxOverlap(track.prediction, prediction) }))
      .filter((candidate) => candidate.overlap >= 0.25)
      .sort((left, right) => right.overlap - left.overlap)[0];
    if (match) {
      match.track.lastSeen = timestampSeconds;
      match.track.prediction = prediction;
      seenTrackIds.add(match.track.id);
      continue;
    }
    const track = { id: nextTrackId++, label: prediction.label, prediction, lastSeen: timestampSeconds };
    activeViolationTracks.push(track);
    seenTrackIds.add(track.id);
    if (byId("save-evidence").checked) queueEvidenceSave(prediction, frame, runSourceLabel);
  }
}

function queueEvidenceSave(prediction, frame, source) {
  const evidence = cropAsDataUrl(prediction, frame);
  if (!evidence) {
    evidenceSaveErrors += 1;
    return;
  }
  const task = byId("task-select").value;
  const save = apiRequest("/api/events", {
    method: "POST",
    body: JSON.stringify({
      task,
      class_name: prediction.label,
      confidence: prediction.confidence,
      source,
      evidence,
    }),
  }).then(() => loadEvents()).catch((error) => {
    evidenceSaveErrors += 1;
    showAlert(`A violation crop could not be saved: ${error.message}`);
  });
  pendingEvidenceSaves.push(save);
}

function updateRunMetrics(predictions) {
  const task = byId("task-select").value;
  const confirmed = predictions.filter((prediction) => prediction.confidence >= CONFIDENCE_THRESHOLD);
  totalDetections += predictions.length;
  totalViolations += confirmed.filter((prediction) => isViolation(task, prediction.label)).length;
  totalCompliant += confirmed.filter((prediction) => isCompliant(task, prediction.label)).length;
  processedFrames += 1;
  byId("metric-total").textContent = String(totalDetections);
  byId("metric-violations").textContent = String(totalViolations);
  byId("metric-compliant").textContent = String(totalCompliant);
  renderPredictions(predictions);
  byId("result-note").textContent = `${processedFrames} frame${processedFrames === 1 ? "" : "s"} analyzed. Counts are detection events across sampled frames; only detections at or above 95% are confirmed.`;
}

function resetRunMetrics() {
  totalDetections = 0;
  totalViolations = 0;
  totalCompliant = 0;
  processedFrames = 0;
  activeViolationTracks = [];
  nextTrackId = 1;
  latestPredictions = [];
  evidenceSaveErrors = 0;
  pendingEvidenceSaves = [];
  byId("metric-total").textContent = "0";
  byId("metric-violations").textContent = "0";
  byId("metric-compliant").textContent = "0";
  byId("prediction-list").replaceChildren();
  byId("video-download").hidden = true;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
}

function showResultPanel() {
  byId("result-empty").hidden = true;
  byId("result-content").hidden = false;
}

function setRunProgress(label, percent = 0) {
  byId("process-progress").hidden = false;
  byId("progress-label").textContent = label;
  byId("progress-percent").textContent = `${Math.round(percent)}%`;
  byId("progress-bar").value = percent;
}

function startVideoRecording(token) {
  if (typeof resultCanvas.captureStream !== "function" || typeof MediaRecorder === "undefined") {
    byId("result-note").textContent = "Frame analysis is running. This browser does not support creating an annotated video download.";
    return;
  }
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/mp4"];
  recordingType = candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  if (!recordingType) {
    byId("result-note").textContent = "Frame analysis is running. This browser cannot record an annotated video download.";
    return;
  }
  try {
    const stream = resultCanvas.captureStream(15);
    mediaRecorder = new MediaRecorder(stream, { mimeType: recordingType });
    recordingChunks = [];
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) recordingChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      for (const track of stream.getTracks()) track.stop();
      if (token !== runToken || !recordingChunks.length) return;
      const blob = new Blob(recordingChunks, { type: recordingType });
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      downloadUrl = URL.createObjectURL(blob);
      const link = byId("video-download");
      link.href = downloadUrl;
      link.download = `${chosenVideoFile?.name.replace(/\.[^.]+$/, "") || "safety-analysis"}-annotated.${recordingType.includes("mp4") ? "mp4" : "webm"}`;
      link.hidden = false;
    });
    mediaRecorder.start(1000);
  } catch (error) {
    showAlert(`Video analysis will continue, but recording is unavailable: ${error.message}`);
  }
}

function startRun(mode, sourceLabel) {
  runToken += 1;
  runMode = mode;
  runSourceLabel = sourceLabel;
  processingStopped = false;
  finishingRun = false;
  analysisPending = false;
  lastAnalysisAt = performance.now() - FRAME_INTERVAL_MS;
  resetRunMetrics();
  showResultPanel();
  byId("media-preview").hidden = false;
  byId("media-preview").classList.remove("stopped");
  byId("media-status-label").textContent = mode === "video" ? "PLAYING · ANALYZING" : "LIVE · ANALYZING";
  byId("media-preview-time").textContent = "0:00";
  runMediaStartedAt = performance.now();
  byId("source-video").controls = mode === "video";
  byId("task-select").disabled = true;
  byId("result-badge").textContent = mode === "video" ? "VIDEO PROCESSING" : "WEBCAM ACTIVE";
  byId("result-badge").className = "result-badge ready";
  byId("video-download").hidden = true;
  byId("process-stop").hidden = mode !== "video";
  if (mode === "video") byId("video-start").disabled = true;
  const token = runToken;
  if (mode === "video") {
    drawImage([], sourceVideo);
    startVideoRecording(token);
    setRunProgress("Video processing · 1 frame / 1.5 sec", 0);
  } else {
    setRunProgress("Webcam active · 1 frame / 1.5 sec", 0);
  }
  animationFrame = requestAnimationFrame(() => processFrameLoop(token));
}

async function processSampledFrame(source, token) {
  const frame = prepareFrame(source);
  const predictions = await requestPredictions(frame, byId("task-select").value);
  if (token !== runToken || processingStopped) return;
  latestPredictions = predictions;
  drawImage(predictions, frame);
  updateRunMetrics(predictions);
  const timestamp = runMode === "video" ? source.currentTime : (Date.now() - runStartedAt) / 1000;
  trackAndSaveViolations(predictions, frame, timestamp);
  if (runMode === "video") {
    const percent = source.duration > 0 ? Math.min(100, source.currentTime / source.duration * 100) : 0;
    setRunProgress(`Video · ${processedFrames} sampled frame${processedFrames === 1 ? "" : "s"}`, percent);
  } else {
    setRunProgress(`Webcam active · ${processedFrames} sampled frame${processedFrames === 1 ? "" : "s"}`, 0);
  }
}

let runStartedAt = 0;
let runMediaStartedAt = 0;

function processFrameLoop(token) {
  if (token !== runToken || !runMode) return;
  const source = sourceVideo;
  const videoEnded = runMode === "video" && source.ended;
  if (!videoEnded && source.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawImage(latestPredictions, source);
    const elapsed = runMode === "video" ? source.currentTime : (performance.now() - runMediaStartedAt) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60).toString().padStart(2, "0");
    byId("media-preview-time").textContent = `${minutes}:${seconds}`;
  }
  if (runMode === "video" && Number.isFinite(source.duration) && source.duration > 0) {
    const percent = Math.min(100, source.currentTime / source.duration * 100);
    byId("progress-bar").value = percent;
    byId("progress-percent").textContent = `${Math.round(percent)}%`;
  }
  if ((processingStopped || videoEnded) && !analysisPending) {
    finishRun(token, videoEnded);
    return;
  }
  const now = performance.now();
  if (!processingStopped && !videoEnded && !analysisPending
      && source.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      && now - lastAnalysisAt >= FRAME_INTERVAL_MS) {
    analysisPending = true;
    lastAnalysisAt = now;
    processSampledFrame(source, token)
      .catch((error) => showAlert(`Frame analysis failed: ${error.message}`))
      .finally(() => { analysisPending = false; });
  }
  animationFrame = requestAnimationFrame(() => processFrameLoop(token));
}

function finishRun(token, completed) {
  if (token !== runToken || finishingRun) return;
  finishingRun = true;
  cancelAnimationFrame(animationFrame);
  if (runMode === "video") sourceVideo.pause();
  if (runMode === "webcam" && sourceVideo.srcObject) {
    for (const track of sourceVideo.srcObject.getTracks()) track.stop();
    sourceVideo.srcObject = null;
    cameraPreview.srcObject = null;
    cameraPreview.hidden = true;
    byId("camera-preview-empty").hidden = false;
    byId("camera-preview-empty").textContent = "Camera stopped. Start webcam to view it again.";
    byId("camera-preview-status").textContent = "CAMERA STOPPED";
    byId("camera-preview-frame").classList.remove("live");
  }
  sourceVideo.controls = false;
  byId("media-status-label").textContent = runMode === "video"
    ? completed ? "VIDEO COMPLETE" : "VIDEO PAUSED"
    : "WEBCAM STOPPED";
  byId("media-preview").classList.add("stopped");
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  mediaRecorder = null;
  byId("camera-start").hidden = false;
  byId("camera-stop").hidden = true;
  byId("task-select").disabled = false;
  byId("video-start").disabled = !chosenVideoFile;
  byId("process-stop").hidden = true;
  byId("result-badge").textContent = completed ? "VIDEO COMPLETE" : "PROCESSING STOPPED";
  byId("result-badge").className = "result-badge ready";
  byId("progress-label").textContent = completed
    ? `Video complete · ${processedFrames} sampled frames`
    : `${runMode === "webcam" ? "Webcam stopped" : "Video stopped"} · ${processedFrames} sampled frames`;
  if (completed) {
    byId("progress-bar").value = 100;
    byId("progress-percent").textContent = "100%";
  }
  byId("result-note").textContent = `${processedFrames} sampled frames analyzed. Counts are detection events across sampled frames.`
    + (evidenceSaveErrors ? ` ${evidenceSaveErrors} evidence crop(s) could not be saved.` : "");
  if (runMode === "video" && !completed && !byId("video-download").hidden) {
    byId("video-download").setAttribute("download", "safety-analysis-partial.webm");
  }
  const saves = pendingEvidenceSaves.slice();
  runMode = null;
  Promise.allSettled(saves).then(() => loadEvents());
}

function stopActiveSources() {
  runToken += 1;
  processingStopped = true;
  cancelAnimationFrame(animationFrame);
  if (sourceVideo.srcObject) {
    for (const track of sourceVideo.srcObject.getTracks()) track.stop();
    sourceVideo.srcObject = null;
  }
  cameraPreview.pause();
  cameraPreview.srcObject = null;
  cameraPreview.hidden = true;
  byId("camera-preview-empty").hidden = false;
  byId("camera-preview-empty").textContent = "Your live camera preview appears here";
  byId("camera-preview-status").textContent = "CAMERA OFF";
  byId("camera-preview-frame").classList.remove("live");
  sourceVideo.pause();
  sourceVideo.controls = false;
  byId("media-preview").hidden = true;
  byId("media-preview").classList.remove("stopped");
  if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  mediaRecorder = null;
  runMode = null;
  analysisPending = false;
  finishingRun = false;
  byId("task-select").disabled = false;
}

byId("video-start").addEventListener("click", async () => {
  if (!chosenVideoFile || !videoUrl) return;
  showAlert("");
  try {
    sourceVideo.srcObject = null;
    sourceVideo.src = videoUrl;
    sourceVideo.muted = true;
    sourceVideo.loop = false;
    sourceVideo.load();
    await new Promise((resolve, reject) => {
      if (sourceVideo.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();
      sourceVideo.addEventListener("loadedmetadata", resolve, { once: true });
      sourceVideo.addEventListener("error", () => reject(new Error("The selected video could not be decoded by this browser.")), { once: true });
    });
    if (!Number.isFinite(sourceVideo.duration) || sourceVideo.duration > 120) {
      throw new Error("Video clips must have a readable duration of 2 minutes or less.");
    }
    sourceVideo.currentTime = 0;
    await sourceVideo.play();
    runStartedAt = Date.now();
    startRun("video", `video: ${chosenVideoFile.name.slice(0, 170)}`);
  } catch (error) {
    sourceVideo.pause();
    showAlert(error.message);
  }
});

byId("camera-start").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    showAlert("Webcam access requires a supported browser and a secure HTTPS connection.");
    return;
  }
  showAlert("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    sourceVideo.pause();
    sourceVideo.removeAttribute("src");
    sourceVideo.load();
    sourceVideo.srcObject = stream;
    sourceVideo.muted = true;
    cameraPreview.srcObject = stream;
    cameraPreview.hidden = false;
    byId("camera-preview-empty").hidden = true;
    byId("camera-preview-frame").classList.add("live");
    byId("camera-preview-status").textContent = "LIVE CAMERA";
    await cameraPreview.play();
    await sourceVideo.play();
    for (const track of stream.getVideoTracks()) {
      track.addEventListener("ended", () => {
        if (runMode === "webcam") {
          processingStopped = true;
          showAlert("The webcam stream ended.");
        }
      }, { once: true });
    }
    runStartedAt = Date.now();
    startRun("webcam", "webcam");
    byId("camera-start").hidden = true;
    byId("camera-stop").hidden = false;
    byId("camera-help").textContent = "Camera is active. Sampled frames go to the inference API every 1.5 seconds.";
  } catch (error) {
    if (sourceVideo.srcObject) {
      for (const track of sourceVideo.srcObject.getTracks()) track.stop();
      sourceVideo.srcObject = null;
    }
    cameraPreview.srcObject = null;
    cameraPreview.hidden = true;
    byId("camera-preview-empty").hidden = false;
    byId("camera-preview-frame").classList.remove("live");
    byId("camera-preview-status").textContent = "CAMERA OFF";
    showAlert(`Could not start the webcam: ${error.message}`);
  }
});

byId("camera-stop").addEventListener("click", () => {
  processingStopped = true;
  byId("camera-help").textContent = "Camera stopped. Start webcam to analyze another session.";
});

byId("process-stop").addEventListener("click", () => {
  processingStopped = true;
  if (runMode === "video") sourceVideo.pause();
});

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
    const predictions = await requestPredictions(currentImage, task);
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
        const evidence = cropAsDataUrl(prediction, currentImage);
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
