(async function(){
    const startBtn = document.getElementById('startBtn');
    const usernameInput = document.getElementById('username');
    const startScreen = document.getElementById('start-screen');
    const examScreen = document.getElementById('exam-screen');
    const questionArea = document.getElementById('questionArea');
    const timerEl = document.getElementById('timeLeft');
    const resultScreen = document.getElementById('result-screen');
    const summary = document.getElementById('summary');

    let questionData = await (await fetch('/api/questions')).json();
    if (!questionData.questions || !questionData.questions.length) {
        alert('No Questions Set By Admin.');
        return;
    }
    
    const questions = questionData.questions;

    let currentIndex = 0;
    let username = '';
    let questionRemaining = 0; // Time remaining for current question
    let questionTimerId = null;
    let mediaStream = null;
    let recorder = null;
    let recordedChunks = [];

    startBtn.addEventListener('click', startExam);

    async function startExam(){
        username = usernameInput.value.trim();
        if (!username) return alert('Please enter your name');

        startScreen.style.display = 'none';
        examScreen.style.display = 'block';

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        } catch (e){
            alert('Camera access required.');
            return;
        }

        showQuestion(0);
    }

    function showQuestion(i){
        if (i >= questions.length) return finishExam('Completed');
        currentIndex = i;
        const q = questions[i];
        questionArea.innerHTML = '';

        // Stop any existing timer
        if (questionTimerId) {
            clearInterval(questionTimerId);
        }

        // Set up time for this question (default to 60 seconds if not set)
        questionRemaining = q.timeSeconds || 60;
        timerEl.textContent = questionRemaining;

        const qDiv = document.createElement('div');
        qDiv.innerHTML = `
            <h3>Question ${i+1} of ${questions.length}</h3>
            <p>${escapeHtml(q.text)}</p>
            <div style="margin: 10px 0; padding: 8px; background: #f0f8ff; border-radius: 4px;">
                <small>Time limit: ${q.timeSeconds || 60} seconds</small>
            </div>
        `;

        q.choices.forEach((c, idx) => {
            const btn = document.createElement('button');
            btn.textContent = (idx+1)+'. '+c;
            btn.style.display = 'block';
            btn.style.marginBottom = '6px';
            btn.style.padding = '8px';
            btn.style.width = '100%';
            btn.style.textAlign = 'left';
            btn.addEventListener('click', ()=>selectAnswer(idx));
            qDiv.appendChild(btn);
        });

        //video preview
        const preview = document.createElement('video');
        preview.autoplay = true;
        preview.muted = true;
        preview.playsInline = true;
        preview.width = 320;
        preview.style.margin = '10px 0';
        preview.srcObject = mediaStream;
        qDiv.appendChild(preview);

        questionArea.appendChild(qDiv);

        // Start the timer for this question
        questionTimerId = setInterval(() => {
            questionRemaining -= 1;
            timerEl.textContent = questionRemaining;
            
            // Add visual warning when time is running low
            if (questionRemaining <= 10) {
                timerEl.style.color = 'red';
                timerEl.style.fontWeight = 'bold';
            } else if (questionRemaining <= 30) {
                timerEl.style.color = 'orange';
            } else {
                timerEl.style.color = 'black';
                timerEl.style.fontWeight = 'normal';
            }
            
            if (questionRemaining <= 0) {
                clearInterval(questionTimerId);
                autoAdvanceQuestion();
            }
        }, 1000);

        startRecording();
    }

    async function autoAdvanceQuestion() {
        // Time's up - treat as incorrect answer and move to next question
        if (recorder && recorder.state !== 'inactive') {
            await stopRecorderAndUpload(false); // false = incorrect/timeout
        }
        
        setTimeout(() => {
            showQuestion(currentIndex + 1);
        }, 500); // Small delay to show time expired
    }

    function startRecording(){
        recordedChunks = [];
        try {
            recorder = new MediaRecorder(mediaStream, {mimeType: 'video/webm; codecs=vp9' });
        }catch (e) {
            recorder = new MediaRecorder(mediaStream);
        }
        recorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
        recorder.start();
    }

    async function selectAnswer(selectedIdx) {
        // Clear the timer since user answered
        if (questionTimerId) {
            clearInterval(questionTimerId);
        }

        const q = questions[currentIndex];
        const isCorrect = selectedIdx === Number(q.correctIndex);

        if (recorder && recorder.state !== 'inactive') {
            await stopRecorderAndUpload(isCorrect);
        }
        showQuestion(currentIndex+1);
    }

    function stopRecorderAndUpload(isCorrect){
        return new Promise(resolve => {
            recorder.onstop = async () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm'});
                const fd = new FormData();
                fd.append('username', username);
                fd.append('questionIndex', String(currentIndex+1));
                fd.append('result', isCorrect ? 'Correct' : 'Wrong');
                fd.append('video', blob, 'clip.webm');

                try {
                    await fetch('/api/upload-video', { method:'POST', body: fd});
                } catch(e){ console.error('upload failed', e); }
                resolve();
            };
            recorder.stop();
        })
    }

    function finishExam(reason){
        if (recorder && recorder.state !== 'inactive') recorder.stop();
        if (mediaStream) mediaStream.getTracks().forEach(t=>t.stop());
        if (questionTimerId) clearInterval(questionTimerId);

        examScreen.style.display = 'none';
        resultScreen.style.display = 'block';
        
        const totalQuestions = questions.length;
        const completionMessage = reason === 'Completed' 
            ? `You completed all ${totalQuestions} questions!`
            : `Exam finished (${reason})`;
            
        summary.innerHTML = `
            <p>${completionMessage}</p>
            <p>Your videos were uploaded for review.</p>
            <p>Thank you for taking the exam!</p>
        `;
    }

    function escapeHtml(s){
        return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
})();