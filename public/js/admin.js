// public/js/admin.js

let questions = [];
let currentQuestionIndex = null;
let examDuration = 60;

// Render question list in the left panel
function renderQuestionList() {
  const list = document.getElementById("questionList");
  list.innerHTML = "";

  questions.forEach((q, idx) => {
    const btn = document.createElement("button");
    btn.textContent = "Question " + (idx + 1);
    btn.onclick = () => editQuestion(idx);
    list.appendChild(btn);
  });
}

// Create a blank question
function createQuestion() {
  return { text: "", choices: ["", "", "", ""], correct: 0 };
}

// Add new question
document.getElementById("addQuestion").onclick = () => {
  questions.push(createQuestion());
  renderQuestionList();
};

// Edit a question
function editQuestion(index) {
  currentQuestionIndex = index;
  const q = questions[index];

  const editor = document.getElementById("editor");
  editor.innerHTML = `
    <label>Question Text:</label><br>
    <textarea id="qText" rows="3" cols="50">${q.text}</textarea><br><br>

    <label>Choices:</label><br>
    ${q.choices.map((c, i) => `
      <input type="text" id="choice${i}" value="${c}" placeholder="Choice ${i+1}">
      <input type="radio" name="correct" ${q.correct === i ? "checked" : ""} value="${i}"> Correct<br>
    `).join("")}

    <button id="saveQuestion">Save Question</button>
  `;

  document.getElementById("saveQuestion").onclick = saveCurrentQuestion;
}

// Save current question
function saveCurrentQuestion() {
  if (currentQuestionIndex === null) return;

  const text = document.getElementById("qText").value;
  const choices = [];
  let correct = 0;

  for (let i = 0; i < 4; i++) {
    choices[i] = document.getElementById(`choice${i}`).value;
    const radio = document.querySelector(`input[name='correct'][value='${i}']`);
    if (radio.checked) correct = i;
  }

  questions[currentQuestionIndex] = { text, choices, correct };
  alert(`Question ${currentQuestionIndex + 1} saved!`);
}

// Save all questions + duration
document.getElementById("saveAll").onclick = async () => {
  const durationInput = document.getElementById("totalSeconds").value;
  if (durationInput) examDuration = parseInt(durationInput, 10);

  const payload = { questions, examDuration };

  try {
    const res = await fetch("/save-questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    alert(data.message || "Saved successfully!");
  } catch (err) {
    console.error("Error saving questions:", err);
    alert("Failed to save questions!");
  }
};

// Refresh video list
// Refresh video list
document.getElementById("refreshVideos").onclick = async () => {
  try {
    const res = await fetch("/api/videos/list");
    if (!res.ok) throw new Error("Failed to fetch video list");
    const data = await res.json();

    const tree = document.getElementById("videosTree");
    tree.innerHTML = "";

    data.forEach(user => {
      const userDiv = document.createElement("div");
      userDiv.innerHTML = `<h4>${user.name}</h4>`;

      if (user.children && user.children.length > 0) {
        user.children.forEach(file => {
          if (file.type === "file") {
            const link = document.createElement("a");
            link.href = `/videos/${user.name}/${file.name}`;
            link.textContent = file.name;
            link.target = "_blank";
            userDiv.appendChild(link);
            userDiv.appendChild(document.createElement("br"));
          }
        });
      }

      tree.appendChild(userDiv);
    });
  } catch (err) {
    console.error("Error loading videos:", err);
    alert("Failed to load videos");
  }
};



// Init with 10 questions
for (let i = 0; i < 10; i++) {
  questions.push(createQuestion());
}
renderQuestionList();
