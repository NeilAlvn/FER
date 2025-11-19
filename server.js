const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const favicon = require('serve-favicon');
const session = require('express-session');

const app = express();
const PORT = process.env.PORT || 3000;
const VIDEOS_DIR = path.join(__dirname, 'videos');
const EXTRACTED_DIR = path.join(__dirname, 'extracted');

const QUESTIONS_PATH = path.join(process.cwd(), 'questions.json');

// Auto-create directories if missing
if (!fs.existsSync(QUESTIONS_PATH)) {
  fs.writeFileSync(QUESTIONS_PATH, JSON.stringify({ totalSeconds: 60, questions: [] }, null, 2));
  console.log("Created questions.json at:", QUESTIONS_PATH);
}

if (!fs.existsSync(EXTRACTED_DIR)) {
  fs.mkdirSync(EXTRACTED_DIR, { recursive: true });
  console.log("Created extracted directory at:", EXTRACTED_DIR);
}

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));
app.use(favicon(path.join(__dirname, "public", "favicon.ico")));

//Simple session-based login
app.use(
  session({
    secret: 'supersecretkey',
    resave: false,
    saveUninitialized: true,
  })
);

if (!fs.existsSync(VIDEOS_DIR)) fs.mkdirSync(VIDEOS_DIR);

// ------------------
// Helper functions
// ------------------

function loadQuestions() {
  if (!fs.existsSync(QUESTIONS_FILE)) return { totalSeconds: 60, questions: [] };
  return JSON.parse(fs.readFileSync(QUESTIONS_FILE));
}

function saveQuestions(data) {
  try {
    fs.writeFileSync(QUESTIONS_FILE, JSON.stringify(data, null, 2));
    console.log("Questions saved successfully to:", QUESTIONS_FILE);
  } catch (err) {
    console.error("Error writing to questions file:", err);
  }
}

function sanitize(name) {
  return name.replace(/[^a-z0-9\- _\.]/gi, '_');
}

// ------------------
// Authentication routes
// ------------------
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (username === 'admin' && password === 'exam3000') {
    req.session.loggedIn = true;
    return res.json({ ok: true });
  }
  res.status(401).json({ ok: false, message: 'Invalid credentials' });
});

app.post('/logout', (req, res) => {
  req.session.destroy(() => res.json({ ok: true }));
});

// Protect /admin.html
app.get('/admin.html', (req, res, next) => {
  if (!req.session.loggedIn) {
    return res.redirect('/login.html');
  }
  next();
});

app.use(express.static(path.join(__dirname, 'public')));
// ------------------
// API routes
// ------------------
app.use(express.json());

// Get questions
app.get('/api/questions', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(QUESTIONS_PATH, 'utf8'));
    res.json(data);
  } catch (err) {
    res.json({ totalSeconds: 60, questions: [] });
  }
});

// Save questions
app.post('/api/questions', (req, res) => {
  try {
    fs.writeFileSync(QUESTIONS_PATH, JSON.stringify(req.body, null, 2));
    res.json({ ok: true });
  } catch (err) {
    console.error('Error saving questions:', err);
    res.status(500).json({ ok: false });
  }
});

// ------------------
// File Upload Handling
// ------------------
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const { username } = req.body;
    if (!username) return cb(new Error('Missing username field'));

    const userDir = path.join(VIDEOS_DIR, sanitize(username));
    if (!fs.existsSync(userDir)) fs.mkdirSync(userDir, { recursive: true });

    cb(null, userDir);
  },
  filename: function (req, file, cb) {
    const { questionIndex, result } = req.body;
    if (!questionIndex || !result) return cb(new Error('Missing fields'));

    const label = result === 'Correct' ? 'right' : 'wrong';
    const filename = `${label}${questionIndex}.mp4`;
    cb(null, filename);
  }
});

const upload = multer({ storage });

app.post('/api/upload-video', upload.single('video'), (req, res) => {
  if (!req.file){
    return res.status(400).json({ ok: false, message: 'No File Uploaded'});
  }

  res.json({
    ok: true,
    path: `/videos/${sanitize(req.body.username)}/${req.file.filename}`
  });
});

// ------------------
// Videos list for admin
// ------------------

app.get("/api/videos", (req, res) => {
  const videosRoot = path.join(__dirname, "videos");

  fs.readdir(videosRoot, { withFileTypes: true }, (err, users) => {
    if (err) return res.status(500).json({ error: "Failed to read videos directory" });

    const result = {};

    users.forEach(userDir => {
      if (userDir.isDirectory()) {
        const userPath = path.join(videosRoot, userDir.name);
        const files = fs.readdirSync(userPath)
          .filter(f => f.endsWith(".mp4"));
        result[userDir.name] = files;
      }
    });

    res.json(result);
  })
});

// ------------------
// Enhanced Emotions Save API - Saves to extracted/emotions.json with merged structure
// ------------------
app.post("/api/save-emotions", (req, res) => {
  try {
    const newData = req.body;
    if (!newData.folder || !newData.video || !newData.segments) {
      return res.status(400).json({ ok: false, message: "Invalid data structure" });
    }

    const emotionsPath = path.join(EXTRACTED_DIR, "emotions.json");
    let existingData = {};

    // Load existing data if file exists
    if (fs.existsSync(emotionsPath)) {
      try {
        const fileContent = fs.readFileSync(emotionsPath, 'utf8');
        existingData = JSON.parse(fileContent);
      } catch (err) {
        console.warn("Could not parse existing emotions.json, starting fresh:", err.message);
        existingData = {};
      }
    }

    // Create a unique key for this video entry
    const videoKey = `${newData.folder}_${newData.video}`;
    
    // Add or update the entry for this video
    existingData[videoKey] = {
      video: newData.video,
      folder: newData.folder,
      totalDuration: newData.totalDuration,
      segments: newData.segments
    };

    // Save the merged data
    fs.writeFileSync(emotionsPath, JSON.stringify(existingData, null, 2));

    console.log(`Emotions data merged and saved to ${emotionsPath}`);
    console.log(`Video key: ${videoKey}`);
    res.json({ ok: true });
  } catch (err) {
    console.error("Error saving emotions.json:", err);
    res.status(500).json({ ok: false, message: "Server error while saving emotions data" });
  }
});

// ------------------
// API to get extracted emotions data
// ------------------
app.get("/api/extracted-emotions", (req, res) => {
  try {
    const emotionsPath = path.join(EXTRACTED_DIR, "emotions.json");
    
    if (!fs.existsSync(emotionsPath)) {
      return res.json({});
    }

    const data = JSON.parse(fs.readFileSync(emotionsPath, 'utf8'));
    res.json(data);
  } catch (err) {
    console.error("Error reading extracted emotions:", err);
    res.status(500).json({ error: "Failed to read extracted emotions data" });
  }
});

app.use('/videos', express.static(VIDEOS_DIR));
app.use('/extracted', express.static(EXTRACTED_DIR));

// ------------------
// Start server
// ------------------
console.log("Serving static files from:", path.join(__dirname, "public"));
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));