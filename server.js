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

const QUESTIONS_FILE = path.join(__dirname, 'questions.json');
const VIDEOS_DIR = path.join(__dirname, 'videos');

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
  fs.writeFileSync(QUESTIONS_FILE, JSON.stringify(data, null, 2));
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

// Get questions
app.get('/api/questions', (req, res) => {
  const data = loadQuestions();
  res.json(data);
});

// Save questions
app.post('/api/questions', (req, res) => {
  const data = req.body;
  if (!data.questions || !Array.isArray(data.questions))
    return res.status(400).json({ ok: false, message: 'Invalid question format' });

  saveQuestions(data);
  res.json({ ok: true });
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

app.get('/api/videos/list', (req, res) => {
  function walk(dir) {
    const items = [];
    fs.readdirSync(dir).forEach((name) => {
      const p = path.join(dir, name);
      const stat = fs.statSync(p);
      if (stat.isDirectory()) items.push({ name, type: 'dir', children: walk(p) });
      else items.push({ name, type: 'file', path: p.replace(__dirname + path.sep, '') });
    });
    return items;
  }

  if (!fs.existsSync(VIDEOS_DIR)) return res.json([]);
  res.json(walk(VIDEOS_DIR));
});

app.use('/videos/', express.static(VIDEOS_DIR));

// ------------------
// Start server
// ------------------
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
