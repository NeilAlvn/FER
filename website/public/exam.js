const questions = [
  { q: "2 + 2 = ?", choices: ["3", "4", "5", "6"], answer: 1 },
  { q: "5 - 3 = ?", choices: ["1", "2", "3", "4"], answer: 1 },
  { q: "10 / 2 = ?", choices: ["2", "4", "5", "6"], answer: 2 },
  { q: "3 * 3 = ?", choices: ["6", "8", "9", "12"], answer: 2 },
  { q: "7 + 1 = ?", choices: ["6", "7", "8", "9"], answer: 2 },
  { q: "Square root of 81?", choices: ["7", "8", "9", "10"], answer: 2 },
  { q: "Capital of France?", choices: ["London", "Berlin", "Paris", "Rome"], answer: 2 },
  { q: "12 * 12 = ?", choices: ["124", "144", "154", "164"], answer: 1 },
  { q: "Water chemical formula?", choices: ["CO2", "H2O", "O2", "NaCl"], answer: 1 },
  { q: "5^2 = ?", choices: ["10", "15", "20", "25"], answer: 3 },
];

let studentNumber = null;
let currentItem = 0;
let mediaRecorder, recordedChunks = [];
let timeLimit = 15; // default
let timerInterval;
let timeLeft;

document.getElementById("startExam").addEventListener("click", async () => {
  studentNumber = document.getElementById("studentNumber").value;
  timeLimit = parseInt(document.getElementById("timeLimit").value);

  if (!studentNumber || !timeLimit) return;

  const res = await fetch("/check-student", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ studentNumber }),
  });
  const data = await res.json();
  if (data.exists) {
    document.getElementById("errorMsg").textContent = "Student number already exists!";
    return;
  }

  document.getElementById("student-section").style.display = "none";
  document.getElementById("exam-section").style.display = "block";
  startCamera();
  loadQuestion();
});

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  document.getElementById("preview").srcObject = stream;

  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
}

function loadQuestion() {
  if (currentItem >= questions.length) {
    document.getElementById("exam-section").style.display = "none";
    document.getElementById("completed-section").style.display = "block";
    return;
  }

  const q = questions[currentItem];
  document.getElementById("question").textContent = `Item ${currentItem+1}: ${q.q}`;

  const choicesDiv = document.getElementById("choices");
  choicesDiv.innerHTML = "";
  q.choices.forEach((choice, i) => {
    const btn = document.createElement("button");
    btn.textContent = choice;
    btn.onclick = () => finishItem(i === q.answer); // pass correctness
    choicesDiv.appendChild(btn);
  });

  startTimer();
}

function startTimer() {
  clearInterval(timerInterval);
  timeLeft = timeLimit;
  document.getElementById("timer").textContent = timeLeft;

  timerInterval = setInterval(() => {
    timeLeft--;
    document.getElementById("timer").textContent = timeLeft;

    if (timeLeft <= 0) {
      clearInterval(timerInterval);
      // auto mark wrong if no answer chosen
      finishItem(false, true);
    }
  }, 1000);
}

async function finishItem(isCorrect, auto = false) {
  clearInterval(timerInterval);

  recordedChunks = [];
  mediaRecorder.start();

  setTimeout(async () => {
    mediaRecorder.stop();
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const base64Data = await blobToBase64(blob);

      await fetch("/save-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          studentNumber,
          itemNumber: currentItem+1,
          isCorrect,
          videoData: base64Data,
        }),
      });

      document.getElementById("nextBtn").style.display = "block";
      if (auto) {
        document.getElementById("question").textContent += " (TIME UP)";
      }
    };
  }, 3000); // still capture 3s reaction video
}

document.getElementById("nextBtn").addEventListener("click", () => {
  currentItem++;
  document.getElementById("nextBtn").style.display = "none";
  loadQuestion();
});

document.getElementById("continueBtn").addEventListener("click", () => {
  location.reload();
});

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}