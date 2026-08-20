require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const rateLimit = require('express-rate-limit');
const path = require('path');

const app = express();

const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Номи модели Groq дар як constant, то баъд осон иваз карда шавад.
// Модели ҳозира (qwen/qwen3.6-27b) image input-ро дастгирӣ мекунад ва JSON mode дорад.
// Агар ин модел дар вақти иҷро дастгирии image-ро надошта бошад, онро ба
// модели дигари Groq-и vision-дор (масалан meta-llama/llama-4-scout-17b-16e-instruct) иваз кунед.
const GROQ_MODEL = process.env.GROQ_MODEL || 'qwen/qwen3.6-27b';
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

const MAX_FILE_SIZE = 8 * 1024 * 1024; // 8MB
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

/* ---------- SECURITY / MIDDLEWARE ---------- */

// CORS: танҳо барои domain-и худ
const allowedOrigins = (process.env.ALLOWED_ORIGIN || 'http://localhost:3000')
  .split(',')
  .map((o) => o.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      // иҷозат барои request-ҳои бе origin (масалан curl, health check)
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(new Error('CORS: domain иҷозат дода нашудааст'));
    },
  })
);

// Request size limit барои JSON body-ҳо
app.use(express.json({ limit: '1mb' }));

// Basic rate limit барои API endpoint
const analyzeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 дақиқа
  max: 30, // ҳадди аксар 30 request барои ҳар IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Дархостҳои зиёд фиристода шуданд. Лутфан баъдтар дубора кӯшиш кунед.' },
});

// Файлро дар memory нигоҳ медорем (диск не), то баъди коркард фавран тоза шавад
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    if (!ALLOWED_MIME_TYPES.includes(file.mimetype)) {
      return cb(new Error('INVALID_FILE_TYPE'));
    }
    cb(null, true);
  },
});

/* ---------- PROMPT ---------- */

const SYSTEM_PROMPT = `Ту як дастёри таҳлили визуалии акс ҳастӣ. Танҳо аз рӯи он чизе, ки дар акс воқеан намоён аст, таҳлил кун.

Қоидаҳои қатъӣ:
- ҳеҷ гоҳ инсонро аз рӯи зебо/зишт баҳо надеҳ
- синну солро тахмин накун
- миллат, нажод, дин ё дигар хусусиятҳои ҳассосро муайян накун
- беморӣ ё ташхиси тиббӣ нагузор
- натиҷаҳоро ҳамчун visual estimate (тахминӣ) баррасӣ кун, на ҳақиқати мутлақ
- агар чизе аз акс возеҳ муайян нашавад, барои он қимат confidence-и паст деҳ ва дар notes қайд кун
- рӯшноӣ, blur, кунҷ (angle) ва сифати умумии акс (quality)-ро ба назар гир
- агар рӯй дар акс возеҳ намоён набошад, quality ва confidence-ро хеле паст гузор

Ту БОЯД танҳо як объекти JSON-и дуруст баргардонӣ, бидуни ҳеҷ матни изофӣ, бидуни markdown, бидуни backtick. Сохтори он бояд айнан чунин бошад:

{
  "quality": 0,
  "symmetry": 0,
  "dryness": 0,
  "redness": 0,
  "eyeClarity": 0,
  "darkCircles": 0,
  "eyebrows": 0,
  "neck": 0,
  "hair": 0,
  "overall": 0,
  "confidence": 0,
  "notes": []
}

Шарҳи майдонҳо (ҳама integer аз 0 то 100):
- quality: сифати умумии акс (рӯшноӣ, фокус, возеҳӣ)
- symmetry: симметрияи визуалии рӯй
- dryness: 0 = нишонаи хушкии пӯст камтар намоён, 100 = бештар намоён
- redness: 0 = сурхшавии пӯст камтар намоён, 100 = бештар намоён
- eyeClarity: тозагӣ ва равшании намоёни чашм
- darkCircles: 0 = доғи зери чашм камтар намоён, 100 = бештар намоён
- eyebrows: шакл ва зичии намоёни қош
- neck: ҳолати умумии намоёни пӯсти гардан
- hair: саломатии намоёни мӯй (дар акс то куҷое намоён аст)
- overall: баҳои умумии визуалӣ, танҳо дар асоси нишондиҳандаҳои боло
- confidence: эътимоди умумии таҳлил (0-100) — агар акс камкифоя, blur ё рӯй пинҳон бошад, пасттар гузор
- notes: array-и то 3 ҷумлаи кӯтоҳи тоҷикӣ бо мушоҳидаҳои визуалии нейтралӣ (бе баҳои зебо/зишт, бе ташхис)

Ягон матни дигар нанавис — танҳо ин JSON.`;

/* ---------- HELPERS ---------- */

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function sanitizeResult(raw) {
  const numFields = [
    'quality',
    'symmetry',
    'dryness',
    'redness',
    'eyeClarity',
    'darkCircles',
    'eyebrows',
    'neck',
    'hair',
    'overall',
    'confidence',
  ];

  const result = {};
  for (const field of numFields) {
    const val = raw[field];
    if (typeof val !== 'number' || Number.isNaN(val)) {
      return null; // JSON-и нодуруст
    }
    result[field] = clamp(Math.round(val), 0, 100);
  }

  if (!Array.isArray(raw.notes)) {
    result.notes = [];
  } else {
    result.notes = raw.notes
      .filter((n) => typeof n === 'string')
      .slice(0, 5)
      .map((n) => n.slice(0, 300));
  }

  return result;
}

function extractJson(text) {
  if (typeof text !== 'string') return null;
  let cleaned = text.trim();
  // тоза кардани ```json ... ``` fence-ҳо, агар модел сарфи назар аз дастур илова карда бошад
  cleaned = cleaned.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/```\s*$/i, '');
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    // кӯшиши охирин: аввалин {...} блокро дар матн ёбем
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (e2) {
        return null;
      }
    }
    return null;
  }
}

/* ---------- ROUTES ---------- */

app.post('/api/analyze-face', analyzeLimiter, (req, res) => {
  upload.single('image')(req, res, async function (err) {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: 'Ҳаҷми акс аз 8MB зиёд аст.' });
      }
      if (err.message === 'INVALID_FILE_TYPE') {
        return res.status(400).json({ error: 'Формати файл дастгирӣ намешавад. Танҳо JPG, PNG ё WEBP.' });
      }
      return res.status(400).json({ error: 'Акс дуруст фиристода нашуд.' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'Ягон акс фиристода нашуд.' });
    }

    if (!GROQ_API_KEY) {
      console.error('GROQ_API_KEY танзим нашудааст.');
      return res.status(500).json({ error: 'Хидмати таҳлил ҳоло дастрас нест. Лутфан баъдтар дубора кӯшиш кунед.' });
    }

    try {
      const base64Image = req.file.buffer.toString('base64');
      const dataUrl = `data:${req.file.mimetype};base64,${base64Image}`;

      const groqResponse = await fetch(GROQ_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${GROQ_API_KEY}`,
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          temperature: 0.3,
          max_tokens: 800,
          response_format: { type: 'json_object' },
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            {
              role: 'user',
              content: [
                { type: 'text', text: 'Ин акси рӯйро тибқи дастур таҳлил кун ва танҳо JSON баргардон.' },
                { type: 'image_url', image_url: { url: dataUrl } },
              ],
            },
          ],
        }),
      });

      // Акс дигар лозим нест — reference-ро озод мекунем (memory-и server дар охири
      // request GC мешавад, ба диск ҳеҷ гоҳ навишта нашудааст)
      req.file.buffer = null;

      if (!groqResponse.ok) {
        const errText = await groqResponse.text().catch(() => '');
        console.error('Groq API error:', groqResponse.status, errText);
        return res.status(502).json({ error: 'Таҳлил ҳоло иҷро нашуд. Лутфан баъдтар дубора кӯшиш кунед.' });
      }

      const data = await groqResponse.json();
      const content = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;

      const parsed = extractJson(content);
      if (!parsed) {
        console.error('Groq JSON parse failed. Raw content:', content);
        return res.status(500).json({ error: 'Натиҷаи AI дуруст хонда нашуд' });
      }

      const sanitized = sanitizeResult(parsed);
      if (!sanitized) {
        console.error('Groq JSON sanitize failed. Parsed:', parsed);
        return res.status(500).json({ error: 'Натиҷаи AI дуруст хонда нашуд' });
      }

      return res.json(sanitized);
    } catch (error) {
      console.error('Analyze-face error:', error);
      return res.status(500).json({ error: 'Таҳлил ҳоло иҷро нашуд. Лутфан баъдтар дубора кӯшиш кунед.' });
    }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ ok: true, model: GROQ_MODEL, hasApiKey: Boolean(GROQ_API_KEY) });
});

// Хидматрасонии frontend (single HTML file)
app.use(express.static(path.join(__dirname), { index: 'index.html' }));

app.listen(PORT, () => {
  console.log(`Rӯйи Ман AI сервер дар http://localhost:${PORT} кор карда истодааст`);
  if (!GROQ_API_KEY) {
    console.warn('⚠️  GROQ_API_KEY дар .env танзим нашудааст — /api/analyze-face кор намекунад.');
  }
});
