import express from 'express';
import multer from 'multer';
import cors from 'cors';
import fs from 'fs-extra';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { Client, handle_file } from '@gradio/client';

const app = express();
const PORT = process.env.PORT || 3000;
const UPLOAD_DIR = 'uploads';
const MAX_FILE_SIZE = 50 * 1024 * 1024;
const HUGGINGFACE_SPACE = 'yisol/IDM-VTON';
// ⚠️ PASTE YOUR NEW/ROTATED TOKEN HERE — get one at
// https://huggingface.co/settings/tokens
// Anyone with access to this file (repo, zip, screenshot) gets this token.
// Don't push this file to a public GitHub repo.
const HF_TOKEN = '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';

app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(UPLOAD_DIR));

fs.ensureDirSync(UPLOAD_DIR);
console.log(`Upload directory ready: ${UPLOAD_DIR}`);

const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    await fs.ensureDir(UPLOAD_DIR);
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}_${Date.now()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['image/jpeg', 'image/png', 'image/webp'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPEG, PNG, and WebP are allowed.'));
    }
  }
});

const validateFile = (file) => {
  if (!file) return 'File is required';
  if (file.size > MAX_FILE_SIZE) return 'File size exceeds 50MB limit';
  return null;
};

const cleanupFiles = async (files) => {
  for (const f of files) {
    try {
      if (f?.path) await fs.remove(f.path);
    } catch (err) {
      console.error(`Failed to cleanup ${f.path}:`, err);
    }
  }
};

// Connect lazily/in the background instead of blocking server startup.
// Health checks on hosting platforms (Render, Railway, Back4app) probe the
// port within a few seconds — if we await this connection before app.listen,
// a slow or cold Hugging Face Space can make the platform kill the container
// for "not becoming healthy" before it ever gets a chance to listen.
let gradioClient;
let connectingPromise = null;

const getGradioClient = async () => {
  if (gradioClient) return gradioClient;
  if (!connectingPromise) {
    connectingPromise = Client.connect(HUGGINGFACE_SPACE, HF_TOKEN ? { hf_token: HF_TOKEN } : undefined)
      .then((client) => {
        gradioClient = client;
        console.log('Connected to Hugging Face Space:', HUGGINGFACE_SPACE);
        return client;
      })
      .catch((err) => {
        console.error('Failed to connect to Hugging Face Space:', err);
        connectingPromise = null; // allow retry on next request
        throw err;
      });
  }
  return connectingPromise;
};

app.post('/api/tryon', upload.fields([
  { name: 'person', maxCount: 1 },
  { name: 'cloth', maxCount: 1 }
]), async (req, res) => {
  const uploadedFiles = [];

  try {
    const personFile = req.files?.person?.[0];
    const clothFile = req.files?.cloth?.[0];

    const personError = validateFile(personFile);
    if (personError) return res.status(400).json({ success: false, error: personError });
    const clothError = validateFile(clothFile);
    if (clothError) return res.status(400).json({ success: false, error: clothError });

    uploadedFiles.push(personFile, clothFile);
    console.log(`Processing: person=${personFile.filename}, cloth=${clothFile.filename}`);

    const client = await getGradioClient();

    const result = await client.predict('/tryon', {
      dict: {
        background: handle_file(personFile.path),
        layers: [],
        composite: null
      },
      garm_img: handle_file(clothFile.path),
      garment_des: '',
      is_checked: true,
      is_checked_crop: true,
      denoise_steps: 30,
      seed: 42
    });

    console.log('Prediction completed');

    let imageUrl = null;
    if (result?.data && Array.isArray(result.data) && result.data.length > 0) {
      const output = result.data[0];
      if (typeof output === 'string') {
        imageUrl = output;
      } else if (output?.url) {
        imageUrl = output.url;
      } else if (output?.path) {
        // Newer gradio versions expose a server-relative path; build the
        // full file URL from the connected client's config.
        const base = client.config?.root || `https://${HUGGINGFACE_SPACE.replace('/', '-')}.hf.space`;
        imageUrl = `${base}/file=${output.path}`;
      }
    }

    if (!imageUrl) {
      console.error('No image URL in result:', JSON.stringify(result));
      return res.status(500).json({ success: false, error: 'Failed to extract generated image' });
    }

    await cleanupFiles(uploadedFiles);
    res.json({ success: true, image: imageUrl });

  } catch (error) {
    console.error('TryOn error:', error);
    await cleanupFiles(uploadedFiles);
    res.status(500).json({ success: false, error: error.message || 'Internal server error' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ success: false, error: 'File size exceeds 50MB limit' });
  }
  res.status(500).json({ success: false, error: 'Internal server error' });
});

// Bind to 0.0.0.0 so the platform's router can reach it, and start listening
// immediately so health checks pass right away. The Hugging Face connection
// happens lazily on first request via getGradioClient().
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Using Hugging Face Space: ${HUGGINGFACE_SPACE}`);
  // Warm the connection in the background (non-blocking) so the first real
  // request is faster, but don't let it delay startup or crash the server.
  getGradioClient().catch(() => {});
});
