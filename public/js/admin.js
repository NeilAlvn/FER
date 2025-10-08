// public/js/admin.js

let questions = [];
let currentQuestionIndex = null;
let examDuration = 60;

// Render question list in the left panel
function renderQuestionList() {
  const questionList = document.getElementById('questionList');
  questionList.innerHTML = '';

  const addBtn = document.createElement('button');
  addBtn.textContent = 'Add Question';
  addBtn.onclick = addQuestion;
  questionList.appendChild(addBtn);
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
document.getElementById('saveAll').addEventListener('click', async () => {
  const totalSeconds = Number(document.getElementById('totalSeconds').value) || 60;

  const questions = [];
  document.querySelectorAll('.question-item').forEach(item => {
    const text = item.querySelector('.question-text')?.value?.trim() || '';
    const choices = [...item.querySelectorAll('.choice')].map(c => c.value.trim());
    const correctIndex = Number(item.querySelector('.correct')?.value || 0);
    questions.push({ text, choices, correctIndex });
  });

  console.log('Sending to server:', { totalSeconds, questions }); //  debug line

  try {
    const res = await fetch('/api/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ totalSeconds, questions })
    });

    const data = await res.json();
    console.log('Server response:', data); // 👈 debug line

    if (data.ok) alert(' Questions saved successfully!');
    else alert(' Failed to save questions.');
  } catch (err) {
    console.error('Save error:', err);
    alert('Error saving questions.');
  }
});

// --- Refresh video list with collapsible user folders ---
document.getElementById("refreshVideos").onclick = async () => {
  try {
    const res = await fetch("/api/videos/list");
    if (!res.ok) throw new Error("Failed to fetch video list");
    const data = await res.json();

    const tree = document.getElementById("videosTree");
    tree.innerHTML = "";

    data.forEach(user => {
      // --- Create user header (acts as dropdown button)
      const userDiv = document.createElement("div");
      userDiv.className = "user-folder";

      const header = document.createElement("div");
      header.className = "user-header";
      header.textContent = `${user.name.toUpperCase()}`;
      header.style.cursor = "pointer";
      header.style.fontWeight = "bold";
      header.style.margin = "5px 0";
      header.style.userSelect = "none";

      const fileList = document.createElement("div");
      fileList.className = "user-files";
      fileList.style.display = "none"; // hidden by default
      fileList.style.marginLeft = "15px";

      // --- Add files (videos) under the user
      if (user.children && user.children.length > 0) {
        user.children.forEach(file => {
          if (file.type === "file" && file.name.endsWith(".mp4")) {
            const link = document.createElement("a");
            link.href = `/videos/${user.name}/${file.name}`;
            link.textContent = ` ${file.name}`;
            link.style.display = "block";
            link.style.margin = "2px 0";
            link.style.textDecoration = "none";
            link.style.color = "#007bff";
            link.addEventListener("mouseover", () => link.style.textDecoration = "underline");
            link.addEventListener("mouseout", () => link.style.textDecoration = "none");

            // --- Play video inside the videoPlayer when clicked ---
            link.addEventListener("click", (e) => {
              e.preventDefault();
              const videoSrc = link.getAttribute("href");
              const videoPlayer = document.getElementById("videoPlayer");
              videoPlayer.src = videoSrc;
              videoPlayer.load();
              videoPlayer.play();

              // Highlight active video
              document.querySelectorAll("#videosTree a").forEach(a => a.classList.remove("active"));
              link.classList.add("active");
            });

            fileList.appendChild(link);
          }
        });
      }

      // --- Toggle dropdown ---
      header.addEventListener("click", () => {
        const allLists = document.querySelectorAll(".user-files");
        allLists.forEach(l => {
          if (l !== fileList) l.style.display = "none"; // close others
        });

        // toggle visibility of current user's files
        fileList.style.display = fileList.style.display === "none" ? "block" : "none";
      });

      userDiv.appendChild(header);
      userDiv.appendChild(fileList);
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

// --- Video segmentation and JSON generation logic --- //
document.addEventListener('DOMContentLoaded', () => {
  const questionList = document.getElementById('questionList');
  const editor = document.getElementById('editor');
  const addBtn = document.getElementById('addQuestion');
  const saveAllBtn = document.getElementById('saveAll');
  const totalSecondsInput = document.getElementById('totalSeconds');

  let questions = [];
  let currentIndex = 0;

  // --- load questions from server (or create 10 placeholders) ---
  async function loadQuestionsFromServer() {
    try {
      const res = await fetch('/api/questions');
      const data = await res.json();
      console.log('Loaded question data:', data);

      if (!data || !Array.isArray(data.questions) || data.questions.length === 0) {
        // create 10 blank placeholders if none present
        questions = Array.from({ length: 10 }, () => ({ text: '', choices: ['', '', '', ''], correctIndex: 0 }));
      } else {
        // ensure each question has expected shape
        questions = data.questions.map(q => ({
          text: q.text || '',
          choices: Array.isArray(q.choices) ? q.choices.slice(0, 4).concat(Array(4 - Math.min(4, (q.choices||[]).length)).fill('')) : ['', '', '', ''],
          correctIndex: typeof q.correctIndex === 'number' ? q.correctIndex : (q.correct || 0)
        }));
      }

      totalSecondsInput.value = data.totalSeconds || 60;
      renderQuestionList();
      showQuestion(0);
    } catch (err) {
      console.error('Failed to load questions:', err);
      // fallback: placeholders
      questions = Array.from({ length: 10 }, () => ({ text: '', choices: ['', '', '', ''], correctIndex: 0 }));
      totalSecondsInput.value = 60;
      renderQuestionList();
      showQuestion(0);
    }
  }

  // --- render the left-side list of question buttons ---
  function renderQuestionList() {
    questionList.innerHTML = '';
    questions.forEach((q, idx) => {
      const btn = document.createElement('button');
      btn.textContent = `Question ${idx + 1}`;
      btn.style.display = 'inline-block';
      btn.style.margin = '4px';
      btn.onclick = () => showQuestion(idx);
      questionList.appendChild(btn);
    });
  }

  // --- show an editor for question i (populates editor area) ---
  function showQuestion(i) {
    currentIndex = i;
    const q = questions[i];
    editor.innerHTML = ''; // clear

    // Question label + textarea
    const lbl = document.createElement('label');
    lbl.textContent = 'Question Text:';
    editor.appendChild(lbl);
    editor.appendChild(document.createElement('br'));

    const ta = document.createElement('textarea');
    ta.rows = 4;
    ta.style.width = '100%';
    ta.value = q.text;
    ta.addEventListener('input', (e) => {
      questions[currentIndex].text = e.target.value;
    });
    editor.appendChild(ta);
    editor.appendChild(document.createElement('br'));
    editor.appendChild(document.createElement('br'));

    // Choices
    const choicesLabel = document.createElement('label');
    choicesLabel.textContent = 'Choices:';
    editor.appendChild(choicesLabel);
    editor.appendChild(document.createElement('br'));

    const radioName = `correct-${Date.now()}-${Math.random()}`; // unique name for radio group
    q.choices.forEach((choiceText, idx) => {
      const choiceDiv = document.createElement('div');
      choiceDiv.style.marginBottom = '6px';

      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'choice-input';
      input.value = choiceText;
      input.style.width = '60%';
      input.addEventListener('input', (e) => {
        questions[currentIndex].choices[idx] = e.target.value;
      });

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = radioName;
      radio.value = idx;
      radio.checked = (q.correctIndex === idx);
      radio.style.marginLeft = '10px';
      radio.addEventListener('change', () => {
        questions[currentIndex].correctIndex = idx;
      });

      const radioLabel = document.createElement('span');
      radioLabel.textContent = ' Correct';
      radioLabel.style.marginLeft = '4px';

      choiceDiv.appendChild(input);
      choiceDiv.appendChild(radio);
      choiceDiv.appendChild(radioLabel);

      editor.appendChild(choiceDiv);
    });

    editor.appendChild(document.createElement('br'));

    // Navigation (Prev / Next)
    const navDiv = document.createElement('div');
    const prevBtn = document.createElement('button');
    prevBtn.textContent = '◀ Prev';
    prevBtn.onclick = () => {
      if (currentIndex > 0) {
        showQuestion(currentIndex - 1);
      }
    };
    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next ▶';
    nextBtn.style.marginLeft = '8px';
    nextBtn.onclick = () => {
      if (currentIndex < questions.length - 1) {
        showQuestion(currentIndex + 1);
      }
    };
    navDiv.appendChild(prevBtn);
    navDiv.appendChild(nextBtn);
    navDiv.style.marginTop = '8px';
    editor.appendChild(navDiv);

    // Small hint
    const hint = document.createElement('div');
    hint.style.marginTop = '8px';
    hint.style.fontSize = '12px';
    hint.style.color = '#555';
    hint.textContent = 'Changes autosave locally. Click "Save All" to persist to server.';
    editor.appendChild(hint);
  }

  // --- add question button handler ---
  addBtn.addEventListener('click', () => {
    questions.push({ text: '', choices: ['', '', '', ''], correctIndex: 0 });
    renderQuestionList();
    showQuestion(questions.length - 1);
  });

  // --- save all to server (single button) ---
  saveAllBtn.addEventListener('click', async () => {
    const totalSeconds = Number(totalSecondsInput.value) || 60;

    // Basic validation: ensure we have at least one question with text
    const cleanedQuestions = questions.filter(q => (q.text && q.text.trim()) || q.choices.some(c => c && c.trim()));
    // If none have content, still send full array (you can change behavior)
    const payload = {
      totalSeconds,
      questions: cleanedQuestions.length ? cleanedQuestions : questions
    };

    console.log('Sending to server:', payload);

    try {
      const res = await fetch('/api/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      console.log('Server response:', result);
      if (result.ok) {
        alert('Questions and total seconds saved successfully!');
        // re-render list in case length changed
        renderQuestionList();
      } else {
        alert('Save failed. Check server logs.');
      }
    } catch (err) {
      console.error('Error saving questions:', err);
      alert('Error saving questions. See console for details.');
    }
  });

  // init
  loadQuestionsFromServer();
});