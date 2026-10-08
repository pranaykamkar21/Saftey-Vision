# Saftey Vision

Saftey Vision uses a browser dashboard, Python/OpenCV inference, and Supabase evidence storage. For a live deployment, the frontend and JavaScript API routes run on Vercel while the Python inference API runs separately on Render. Render's free service sleeps after inactivity, so its first request can be slow.

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
2. Deploy the Python inference API using the `render.yaml` blueprint. The inference endpoint is public and does not require an API key. The demo model weights are included in this public repository.
3. Connect this GitHub repository to Vercel. `vercel.json` sets the framework preset to **Other** and the output directory to `public`; leave the build command blank. Vercel serves the static site from `public/` and deploys the JavaScript functions in `api/`.
4. Add the following environment variables in **Vercel → Project → Settings → Environment Variables**, then redeploy:

| Variable | Required | Value |
|---|---|---|
| `INFERENCE_API_URL` | Yes | Render service HTTPS URL followed by `/api/analyze` |
| `INFERENCE_API_KEY` | No | Bearer token for that inference API, if required |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service-role key; keep it server-side |
| `SUPABASE_EVIDENCE_BUCKET` | No | Private bucket name; defaults to `safety-evidence` |

Set `INFERENCE_API_URL` to the Render service URL followed by `/api/analyze`. Leave `INFERENCE_API_KEY` unset. The website and inference API are public: there is no password or sign-in, and anyone can submit requests directly to the inference endpoint. Set environment variables for every Vercel environment you intend to use. Keep the Supabase service-role key in Vercel server-side environment variables; never add it to browser code.

Render's free web service spins down after 15 minutes without traffic, and its filesystem is temporary. Evidence remains in Supabase; the first inference request after a sleep may need a retry after the service wakes. Free compute is intended for testing and may not be reliable enough for production.

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

Supported violation labels are `without_mask`, `mask_worn_incorrectly`, and `no_helmet`; compliant labels can be `with_mask` and `helmet`. The browser confidence slider ranges from 25% to 99%; only violation detections at or above the selected threshold are eligible for evidence capture. Lower thresholds include more uncertain predictions, so review them carefully. The browser crops the returned face/head box before upload; the original photo or video is not stored by this app. The service must use HTTPS and return promptly within the Vercel function timeout.

The public web app supports still photos, video clips, and webcam snapshots. Video files must be MP4, WebM, or browser-supported MOV, no larger than 50 MB and no longer than two minutes. Videos are played and processed locally in the browser; a frame is sent to the inference API every 1.5 seconds. Where browser recording is supported, the annotated video can be downloaded as WebM or MP4; audio is not included. Webcam requires browser permission and HTTPS (provided by Vercel). Start the webcam, capture a still photo, review it, and then choose **Analyze photo**; the live camera is stopped as soon as the snapshot is captured. Video detection counts are sampled-frame events, not unique people.

## Public access, privacy, and limitations

Anyone can submit images or video frames for inference, read and delete evidence records, and upload eligible violation crops. Public access can incur inference and storage costs and lets visitors modify the shared evidence vault; configure provider quotas and Vercel rate limiting before sharing the URL. Photos, sampled video frames, and captured webcam snapshots are sent to the inference service you configure. When evidence capture is enabled, only confirmed violation crops are uploaded to the private Supabase bucket; signed display links expire after an hour, and records/crops older than 30 days are removed when the vault is opened. Deploy only with suitable notice, consent, and a reviewed inference provider. Predictions are advisory and must be reviewed; a missing detection is not proof of compliance.

## Python training and local CLI

The local dashboard and model inference can also be run with `python app.py`. Dataset preparation and training scripts remain local development tools and are not needed by the deployed inference service.
