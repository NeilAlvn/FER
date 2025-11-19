export function initQuestionEditor() {
  console.log("questionEditor initialized");

  const questionList = document.getElementById("questionList");
  const addQuestionBtn = document.getElementById("addQuestion");
  const saveAllBtn = document.getElementById("saveAll");

  if (!questionList || !addQuestionBtn || !saveAllBtn) {
    console.warn("Question editor not initialized: missing elements.");
    return;
  }

  let questions = [];
  let currentIndex = 0;

  async function loadQuestions() {
    try {
      const res = await fetch("/api/questions");
      const data = await res.json();
      questions = data.questions || [];

      if (questions.length === 0) {
        // start with at least 10 questions, each with default 60 seconds
        for (let i = 0; i < 10; i++) {
          questions.push({
            text: `Question ${i + 1}`,
            choices: ["", "", "", ""],
            correctIndex: null,
            timeSeconds: 60 // Default time per question
          });
        }
      } else {
        // Ensure all existing questions have timeSeconds property
        questions = questions.map(q => ({
          ...q,
          timeSeconds: q.timeSeconds || 60 // Default to 60 seconds if not set
        }));
      }

      renderQuestionEditor(currentIndex);
    } catch (err) {
      console.error("Failed to load questions:", err);
    }
  }

  function renderQuestionEditor(index) {
    questionList.innerHTML = "";

    if (!questions[index]) {
      questions[index] = {
        text: `Question ${index + 1}`,
        choices: ["", "", "", ""],
        correctIndex: null,
        timeSeconds: 60
      };
    }

    const q = questions[index];
    const wrapper = document.createElement("div");
    wrapper.classList.add("question-editor");

    wrapper.innerHTML = `
      <div class="question-nav">
        ${questions.map((_, i) => `
          <button class="question-btn ${i === index ? "active" : ""}" data-index="${i}">
            Q${i + 1} (${questions[i].timeSeconds || 60}s)
          </button>
        `).join("")}
        <button id="addQuestionBtn" style="margin-left:10px;">➕ Add Question</button>
      </div>

      <div class="question-field">
        <label><strong>Question ${index + 1}:</strong></label>
        <input type="text" id="questionText" value="${q.text}" placeholder="Enter question text" />
      </div>

      <div class="time-field" style="margin-bottom: 10px;">
        <label><strong>Time Limit (seconds):</strong></label>
        <input type="number" id="questionTime" value="${q.timeSeconds || 60}" min="1" max="600" placeholder="Time in seconds" />
      </div>

      <div class="answers">
        ${q.choices.map((ans, i) => `
          <div class="answer-row">
            <input type="radio" name="correctAnswer" ${q.correctIndex === i ? "checked" : ""} data-index="${i}" />
            <input type="text" class="answerInput" data-index="${i}" value="${ans}" placeholder="Choice ${i + 1}" />
          </div>
        `).join("")}
      </div>

      <div class="controls">
        <button id="prevQuestion">⬅ Previous</button>
        <button id="nextQuestion">Next ➡</button>
        <button id="removeQuestion" style="background:#c0392b; color:#fff; margin-left:10px;">🗑 Remove Question</button>
      </div>
    `;

    questionList.appendChild(wrapper);

    // === Event Listeners ===
    wrapper.querySelectorAll(".question-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        saveCurrentQuestion();
        currentIndex = parseInt(e.target.dataset.index);
        renderQuestionEditor(currentIndex);
      });
    });

    wrapper.querySelectorAll(".answerInput").forEach((input) => {
      input.addEventListener("input", (e) => {
        const idx = parseInt(e.target.dataset.index);
        questions[currentIndex].choices[idx] = e.target.value;
      });
    });

    wrapper.querySelectorAll("input[name='correctAnswer']").forEach((radio) => {
      radio.addEventListener("change", (e) => {
        questions[currentIndex].correctIndex = parseInt(e.target.dataset.index);
      });
    });

    wrapper.querySelector("#questionText").addEventListener("input", (e) => {
      questions[currentIndex].text = e.target.value;
    });

    wrapper.querySelector("#questionTime").addEventListener("input", (e) => {
      const timeValue = parseInt(e.target.value) || 60;
      questions[currentIndex].timeSeconds = timeValue;
      // Update the button text to show new time
      const btn = wrapper.querySelector(`[data-index="${currentIndex}"]`);
      if (btn) {
        btn.textContent = `Q${currentIndex + 1} (${timeValue}s)`;
      }
    });

    wrapper.querySelector("#nextQuestion").addEventListener("click", () => {
      saveCurrentQuestion();
      currentIndex = (currentIndex + 1) % questions.length;
      renderQuestionEditor(currentIndex);
    });

    wrapper.querySelector("#prevQuestion").addEventListener("click", () => {
      saveCurrentQuestion();
      currentIndex = (currentIndex - 1 + questions.length) % questions.length;
      renderQuestionEditor(currentIndex);
    });

    wrapper.querySelector("#addQuestionBtn").addEventListener("click", () => {
      saveCurrentQuestion();
      questions.push({
        text: `Question ${questions.length + 1}`,
        choices: ["", "", "", ""],
        correctIndex: null,
        timeSeconds: 60 // Default 60 seconds for new questions
      });
      currentIndex = questions.length - 1;
      renderQuestionEditor(currentIndex);
    });

    wrapper.querySelector("#removeQuestion").addEventListener("click", () => {
      if (questions.length <= 1) {
        alert("You must have at least one question.");
        return;
      }
      if (confirm(`Remove Question ${index + 1}?`)) {
        questions.splice(index, 1);
        if (currentIndex >= questions.length) currentIndex = questions.length - 1;
        renderQuestionEditor(currentIndex);
      }
    });
  }

  function saveCurrentQuestion() {
    const questionInput = document.getElementById("questionText");
    const timeInput = document.getElementById("questionTime");
    if (questionInput) {
      questions[currentIndex].text = questionInput.value;
    }
    if (timeInput) {
      questions[currentIndex].timeSeconds = parseInt(timeInput.value) || 60;
    }
  }

  async function saveAll() {
    saveCurrentQuestion();
    
    // Calculate total time for all questions
    const totalSeconds = questions.reduce((total, q) => total + (q.timeSeconds || 60), 0);

    const payload = { totalSeconds, questions };

    try {
      const res = await fetch("/api/questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload, null, 2),
      });

      if (res.ok) {
        alert(`Questions saved successfully! Total exam time: ${Math.floor(totalSeconds/60)}m ${totalSeconds%60}s`);
      } else {
        alert("Failed to save questions!");
      }
    } catch (err) {
      console.error("Error saving questions:", err);
      alert("Error saving questions!");
    }
  }

  // Add event listener for main add button
  addQuestionBtn.addEventListener("click", () => {
    saveCurrentQuestion();
    const newIndex = questions.length;
    questions.push({
      text: `Question ${newIndex + 1}`,
      choices: ["", "", "", ""],
      correctIndex: null,
      timeSeconds: 60
    });
    currentIndex = newIndex;
    renderQuestionEditor(currentIndex);
  });

  saveAllBtn.addEventListener("click", saveAll);

  loadQuestions();

  window.getQuestions = () => questions;
}