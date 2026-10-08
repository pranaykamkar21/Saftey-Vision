# Saftey Vision

Saftey Vision is a Python-first dashboard that can run locally with the bundled OpenCV + local model inference stack. The repository also includes a Vercel deployment variant that forwards analysis to an external inference API and stores evidence in Supabase, but the original project behavior is the Python local app.

## Run it locally in Python

1. From the repository root, create and activate a Python environment if needed.
2. Install dependencies from `requirements.txt`.
3. Start the app:

```bash
python app.py
```

This starts the local dashboard at `http://127.0.0.1:4173` and serves the browser UI while running the model in Python. The same local server also exposes the evidence and analysis APIs used by the dashboard.

## Deploy to Vercel

1. Create a Supabase project. In its SQL editor, run [`supabase/schema.sql`](./supabase/schema.sql) to create the private evidence bucket and violations table.
2. Connect this GitHub repository to Vercel. `vercel.json` sets the framework preset to **Other** and the output directory to `public`; leave the build command blank. Vercel serves the static site from `public/` and deploys the JavaScript functions in `api/`; this repository does not contain a Python runtime or Python entrypoint.
3. Add the following environment variables in **Vercel → Project → Settings → Environment Variables**, then redeploy:

| Variable | Required | Value |
|---|---|---|
| `INFERENCE_API_URL` | Yes | HTTPS URL for your image inference API |
| `INFERENCE_API_KEY` | No | Bearer token for that inference API, if required |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service-role key; keep it server-side |
| `SUPABASE_EVIDENCE_BUCKET` | No | Private bucket name; defaults to `safety-evidence` |

The website and its API routes are public: there is no password or sign-in. Set environment variables for every Vercel environment you intend to use. Keep the Supabase service-role key and inference key in Vercel server-side environment variables; never add either to browser code.

## Inference API contract

Vercel sends the configured inference endpoint a `POST` JSON request:

```json
{
  "task": "mask",
  "image": "data:image/jpeg;base64,..."
}
```

If `INFERENCE_API_KEY` is set, Vercel sends it in `Authorization: Bearer ...`. Return JSON with normalized boxes (`x`, `y`, `width`, `height` are fractions from 0 to 1):

```json
{
  "predictions": [
    { "label": "without_mask", "confidence": 0.98, "box": [0.2, 0.1, 0.3, 0.4] }
  ]
}
```

Supported violation labels are `without_mask`, `mask_worn_incorrectly`, and `no_helmet`; compliant labels can be `with_mask` and `helmet`. The browser displays returned predictions, but only violation detections at or above the fixed 95% threshold are eligible for evidence capture. The browser crops the returned face/head box before upload; the original photo or video is not stored by this app. The service must use HTTPS and return promptly within the Vercel function timeout.

The public web app supports still photos, video clips, and webcam snapshots. Video files must be MP4, WebM, or browser-supported MOV, no larger than 50 MB and no longer than two minutes. Videos are played and processed locally in the browser; a frame is sent to the inference API every 1.5 seconds. Where browser recording is supported, the annotated video can be downloaded as WebM or MP4; audio is not included. Webcam requires browser permission and HTTPS (provided by Vercel). Start the webcam, capture a still photo, review it, and then choose **Analyze photo**; the live camera is stopped as soon as the snapshot is captured. Video detection counts are sampled-frame events, not unique people.

## Public access, privacy, and limitations

Anyone can submit images or video frames for inference, read and delete evidence records, and upload eligible violation crops. Public access can incur inference and storage costs and lets visitors modify the shared evidence vault; configure provider quotas and Vercel rate limiting before sharing the URL. Photos, sampled video frames, and captured webcam snapshots are sent to the inference service you configure. When evidence capture is enabled, only confirmed violation crops are uploaded to the private Supabase bucket; signed display links expire after an hour, and records/crops older than 30 days are removed when the vault is opened. Deploy only with suitable notice, consent, and a reviewed inference provider. Predictions are advisory and must be reviewed; a missing detection is not proof of compliance.

## Python training and local CLI

This repository contains only the Vercel web app, JavaScript API routes, and Supabase schema. The Vercel app uses the external inference service configured by `INFERENCE_API_URL`; local Python training scripts and model weights are intentionally excluded from the GitHub deployment source.
