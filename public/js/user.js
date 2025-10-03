(async function(){
    const startBtn = document.getElementById('startBtn');
    const usernameInput = document.getElementById('username');
    const startScreen = document.getElementById('start-screen');
    const examScreen = document.getElementById('exam-screen');
    const questionArea = document.getElementById('questionArea');
    const timerEl = document.getElementById('timeLeft');
    const resultScreen = document.getElementById('result-screen');
    const summary = document.getElementById('summary');

    let questionData = await (await fetch('/questions')).json();
    if (!questionData.questions || !questionData.questions.length) {
        alert('No Questions Set By Admin FUCK U.');
        return;
    }
    const totalSeconds = Number(questionData.totalSeconds || 60);
    const questions = questionData.questions.slice(0,10);

    let currentIndex = 0;
    let username = '';
    let overallRemaining = totalSeconds;
    let overallTimerId = null;
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


        overallTimerId = setInterval(()=>{
            overallRemaining -= 1;
            timerEl.textContent = overallRemaining;
            if (overallRemaining <=0) {
                clearInterval(overallTimerId);
                finishExam('timeup');
            }
        }, 1000);


        timerEl.textContent = overallRemaining;
        showQuestion(0);
    }

    function showQuestion(i){
        if (i >= questions.length) return finishExam('Completed');
        currentIndex = i;
        const q = questions[i];
        questionArea.innerHTML = '';

        const qDiv = document.createElement('div');
        qDiv.innerHTML = `<h3>Question ${i+1}</h3><p>${escapeHtml(q.text)}</p>`;

        q.choices.forEach((c, idx) => {
            const btn = document.createElement('button');
            btn.textContent = (idx+1)+'. '+c;
            btn.style.display = 'block';
            btn.style.marginBottom = '6px';
            btn.addEventListener('click', ()=>selectAnswer(idx));
            qDiv.appendChild(btn);
        });

        //video preview
        const preview = document.createElement('video');
        preview.autoplay = true;
        preview.muted = true;
        preview.playsInline = true;
        preview.width = 320;
        preview.srcObject = mediaStream;
        qDiv.appendChild(preview);

        questionArea.appendChild(qDiv);

        startRecording();
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
                const blob = new Blob(recordedChunks, { type: 'vide/webm'});
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
        clearInterval(overallTimerId);

        examScreen.style.display = 'none';
        resultScreen.style.display = 'block';
        summary.innerHTML = `<p>Exam finished (${reason}). Your videos were uploaded.</p>`;
    }

    function escapeHtml(s){
        return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
})();