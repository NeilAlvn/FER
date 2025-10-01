const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");

const app = express();
const PORT = 3000;

app.use(express.static("public"));
app.use(express.json({ limit: "50mb" }));

// Ensure recordings folder exists
const recordingsDir = path.join(__dirname, "recordings");
if (!fs.existsSync(recordingsDir)) {
    fs.mkdirSync(recordingsDir);
}

// Check if student already exists
app.post("/check-student", (req, res) => {
    const { studentNumber } = req.body;
    const studentPath = path.join(recordingsDir, `student${studentNumber}`);
    if (fs.existsSync(studentPath)) {
        return res.json({ exists: true });
    }
    fs.mkdirSync(studentPath);
    res.json({ exists: false });
});

// Save video per item
app.post("/save-video", (req, res) => {
    const { studentNumber, itemNumber, videoData, isCorrect } = req.body;

    const studentPath = path.join(recordingsDir, `student${studentNumber}`);
    const itemPath = path.join(studentPath, `item${itemNumber}`);
    if (!fs.existsSync(itemPath)) {
        fs.mkdirSync(itemPath);
    }

    const status = isCorrect ? "correct" : "wrong";
    const videoFile = path.join(itemPath, `${status}${studentNumber}-${itemNumber}.webm`);

    const videoBuffer = Buffer.from(videoData.split(",")[1], "base64");
    fs.writeFileSync(videoFile, videoBuffer);

    res.json({ success: true });
});


app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
