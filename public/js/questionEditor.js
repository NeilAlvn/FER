export function initQuestionEditor() {
  console.log("questionEditor initialized");

  const questionList = document.getElementById("questionList");
  const addQuestionBtn = document.getElementById("addQuestion");
  const saveAllBtn = document.getElementById("saveAll");
  const totalSecondsInput = document.getElementById("totalSeconds");

  if (!questionList || !addQuestionBtn || !saveAllBtn) {
    console.warn("Question editor not initialized: missing elements.");
    return;
  }

  let questions = [];
  let currentIndex = 0;
  let totalSeconds = 60;

  // Load existing questions.json
  async function loadQuestions() {
    try {
      const res = await fetch("/api/questions");
      const data = await res.json();
      questions = data.questions || [];
      totalSeconds = data.totalSeconds || 60;
      totalSecondsInput.value = totalSeconds;

      // Ensure always 10 questions
      while (questions.length < 10) {
        questions.push({
          text: `Question ${questions.length + 1}`,
          choices: ["", "", "", ""],
          correctIndex: null,
        });
      }

      renderQuestionEditor(currentIndex);
    } catch (err) {
      console.error("Failed to load questions:", err);
    }
  }

  // Render the question editor UI
  function renderQuestionEditor(index) {
    questionList.innerHTML = "";

    if (!questions[index]) {
      questions[index] = {
        text: `Question ${index + 1}`,
        choices: ["", "", "", ""],
        correctIndex: null,
      };
    }

    const q = questions[index];
    const wrapper = document.createElement("div");
    wrapper.classList.add("question-editor");

    wrapper.innerHTML = `
      <div class="question-nav">
        ${Array.from({ length: 10 }, (_, i) => `
          <button class="question-btn ${i === index ? "active" : ""}" data-index="${i}">
            Q${i + 1}
          </button>
        `).join("")}
      </div>

      <div class="question-field">
        <label><strong>Question ${index + 1}:</strong></label>
        <input type="text" id="questionText" value="${q.text}" placeholder="Enter question text" />
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
        <button id="nextQuestion">Next ➡</button>
      </div>
    `;

    questionList.appendChild(wrapper);

    // Events
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

    wrapper.querySelector("#nextQuestion").addEventListener("click", () => {
      saveCurrentQuestion();
      currentIndex = (currentIndex + 1) % 10;
      renderQuestionEditor(currentIndex);
    });
  }

  // Save currently displayed question
  function saveCurrentQuestion() {
    const questionInput = document.getElementById("questionText");
    if (questionInput) {
      questions[currentIndex].text = questionInput.value;
    }
  }

  // Save all to server
  async function saveAll() {
    saveCurrentQuestion();
    totalSeconds = parseInt(totalSecondsInput.value) || 60;

    const payload = {
      totalSeconds,
      questions,
    };

    try {
      const res = await fetch("/api/questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload, null, 2),
      });

      if (res.ok) {
        alert("Questions saved successfully!");
      } else {
        alert("Failed to save questions!");
      }
    } catch (err) {
      console.error("Error saving questions:", err);
      alert("Error saving questions!");
    }
  }

  saveAllBtn.addEventListener("click", saveAll);
  addQuestionBtn.addEventListener("click", () => renderQuestionEditor(currentIndex));

  loadQuestions();

  // Debug helper
  window.getQuestions = () => questions;
}
