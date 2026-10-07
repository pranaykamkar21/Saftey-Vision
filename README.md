# Saftey Vision

Saftey Vision is a static web dashboard with Vercel serverless API routes. Detection runs on an external inference API; violation records and private face/head crops are stored in Supabase. Model training and local CLI scripts remain Python tools and are not part of the Vercel runtime.

## Deploy to Vercel

1. Create a Supabase project. In its SQL editor, run [`supabase/schema.sql`](./supabase/schema.sql) to create the private evidence bucket and violations table.
2. Connect this GitHub repository to Vercel. Vercel serves the static site from `public/` and deploys the functions in `api/`; no build command or Python runtime is needed.
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

Supported violation labels are `without_mask`, `mask_worn_incorrectly`, and `no_helmet`; compliant labels can be `with_mask` and `helmet`. The browser displays returned predictions, but only violation detections at or above the fixed 95% threshold are eligible for evidence capture. The browser crops the returned face/head box before upload; the original photo is not stored by this app. The service must use HTTPS and return promptly within the Vercel function timeout. The UI currently analyzes still images only; video and live camera processing are not included in this Vercel deployment.

## Public access, privacy, and limitations

Anyone can submit images for inference, read and delete evidence records, and upload eligible violation crops. Public access can incur inference and storage costs and lets visitors modify the shared evidence vault; configure provider quotas and Vercel rate limiting before sharing the URL. Images are sent to the inference service you configure. When evidence capture is enabled, only confirmed violation crops are uploaded to the private Supabase bucket; signed display links expire after an hour, and records/crops older than 30 days are removed when the vault is opened. Deploy only with suitable notice, consent, and a reviewed inference provider. Predictions are advisory and must be reviewed; a missing detection is not proof of compliance.

## Python training and local CLI

The Python requirements and scripts are for local development and model training, not Vercel deployment. Install them with:

```powershell
python -m pip install -r requirements.txt
```

Training, dataset preparation, and local image/video/webcam inference utilities remain under `src/`. Their local SQLite evidence store is separate from the hosted dashboard's Supabase store.
