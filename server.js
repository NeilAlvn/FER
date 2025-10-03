const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const favicon = require("serve-favicon");

const app = express();
const PORT = process.env.PORT || 3000;
const uploadsDir = path.join(__dirname, "uploads");

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json()); //needed for post json
app.use(express.urlencoded({ extended: true }));
app.use(favicon(path.join(__dirname, "public", "favicon.ico")));

// store questions in a JSON file
const QUESTIONS_FILE = path.join(__dirname, 'questions.json');
const VIDEOS_DIR = path.join(__dirname, 'videos');

if (!fs.existsSync(VIDEOS_DIR)) fs.mkdirSync(VIDEOS_DIR);

function loadQuestions() {
    if (!fs.existsSync(QUESTIONS_FILE)) return { totalSeconds: 60, questions: [] };
    return JSON.parse(fs.readFileSync(QUESTIONS_FILE));
}

function savedQuestions(data) {
    fs.writeFileSync(QUESTIONS_FILE, JSON.stringify(data, null, 2));
}

//simple admin auth (hardcoded credentials)
app.post('/api/admin/login', (req, res) => {
    const { username, password } = req.body;
    if (username === 'Admin123' && password === 'examination3000') {
        return res.json({ ok: true});
    }
    returnres.status(401).json({ ok: false, message: 'Invalid Credentials'});
});

app.get('/api/questions', (req,res) => {
    const data = req.body;
    //basic validation
    if (!data.questions || !Array.isArray(data.questions)) return res.status(200).json({ ok: false});
    savedQuestions(data);
    res.json({ ok: true});
});

//multer for handling file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        //destination depends on fields in the request body
        const { username, questionIndex, result } = req.body;
        if (!username || !questionIndex || !result) return cb (new Error('Missing Fields'));

        const userDir = path.join(VIDEOS_DIR, sanitize(username));
        const questionDir = path.join(userDir, `Question${questionIndex}`);
        const resultDir = path.join(questionDir, result === 'Correct' ? 'Correct' : 'Wrong');

        [userDir, questionDir, resultDir].forEach(d => { if (!fs.existsSync(d)) fs.mkdirSync(d); });

        cb(null, resultDir);
    },
    filename: function (req, file, cb) {
        const ts = Date.now();
        const ext = path.extname(file.originalname) || '.web';
        cb(null, `clip-${ts}${ext}`);
    }
});

const upload = multer({ storage: storage });

app.post('/api/upload-video', upload.single('video'), (req, res) => {
    //return the file path for admin to view
    res.json({ ok: true, path: req.file.path.replace(__dirname + path.sep, '')});
});

app.get('/api/videos/list', (req, res) => {
    //returns a directory tree for /videos
    function walk(dir) {
        const items = [];
        fs.readdirSync(dir).forEach(name => {
            const p = path.join(dir, name);
            const stat = fs.statSync(p);
            if (stat.isDirectory()) items.push({ name, type: 'dir', children: walk(p)});
            else items.push({ name, type: 'file', path: p.replace(__dirname + path.sep, '')});
        });
        return items;
    }

    if (!fs.existsSync(VIDEOS_DIR)) return res.json([]);
    res.json(walk(VIDEOS_DIR));
});

//serve saved videos statically
app.use('/videos', express.static(path.join(__dirname + path.sep, '')));

//Temporary in-use memory store
let examData = { questions: [], examDuration: 60 };

//Save Quetions from admin
app.post("/save-questions", (req, res) => {
  if (!req.body.questions || req.body.questions.length === 0) {
    return res.status(400).json({ message: "No questions provided" });
  }

  examData = req.body; // overwrite with new payload
  console.log("Exam data updated:", examData);
  res.json({ message: "Questions saved successfully!" });
});

// Serve questions to user
app.get("/questions", (req, res) => {
  if (!examData.questions || examData.questions.length === 0) {
    return res.json({ message: "No questions set by admin." });
  }
  res.json(examData);
});

app.listen(PORT, () => console.log(`Server started on port ${PORT}`));

//helper: sanitize file/folder names
function sanitize(name){
    return name.replace(/[^a-z0-9\- _\.]/gi, '_');
}

// List all saved videos as JSON
app.get("/videos", (req, res) => {
  if (!fs.existsSync(uploadsDir)) {
    return res.json({ users: [] });
  }

  const users = fs.readdirSync(uploadsDir).map(userFolder => {
    const userPath = path.join(uploadsDir, userFolder);
    if (!fs.statSync(userPath).isDirectory()) return null;

    const results = {};
    ["Correct", "Wrong"].forEach(type => {
      const typePath = path.join(userPath, type);
      if (fs.existsSync(typePath)) {
        results[type] = fs.readdirSync(typePath).map(file => `/uploads/${userFolder}/${type}/${file}`);
      } else {
        results[type] = [];
      }
    });

    return { user: userFolder, results };
  }).filter(Boolean);

  res.json({ users });
});

// Make video files accessible
app.use("/uploads", express.static(uploadsDir));