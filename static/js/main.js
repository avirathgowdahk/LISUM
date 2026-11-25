document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const uploadFormCard = document.getElementById('uploadFormCard');
    const uploadSpinner = document.getElementById('uploadSpinner');
    
    const configCard = document.getElementById('configCard');
    const videoPreviewPlayer = document.getElementById('videoPreviewPlayer');
    const metaFilename = document.getElementById('metaFilename');
    const metaDuration = document.getElementById('metaDuration');
    const metaResolution = document.getElementById('metaResolution');
    const metaAudio = document.getElementById('metaAudio');
    
    const durationChips = document.querySelectorAll('.duration-chip');
    const customDurationBox = document.getElementById('customDurationBox');
    const customDurationInput = document.getElementById('customDurationInput');
    const langChips = document.querySelectorAll('.lang-chip');
    const startSummarizeBtn = document.getElementById('startSummarizeBtn');
    
    const processingCard = document.getElementById('processingCard');
    const progressFill = document.getElementById('progressFill');
    const progressPctText = document.getElementById('progressPctText');
    const statusText = document.getElementById('statusText');

    let uploadedVideoData = null;
    let selectedDurationSec = 120; // default 2 mins
    let selectedLanguage = 'en'; // default English
    let timerInterval = null;
    let startTimeSec = 0;

    // Tab switching elements
    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabUrlBtn = document.getElementById('tabUrlBtn');
    const uploadTabContent = document.getElementById('uploadTabContent');
    const urlTabContent = document.getElementById('urlTabContent');
    const urlInput = document.getElementById('urlInput');
    const fetchUrlBtn = document.getElementById('fetchUrlBtn');
    const urlErrorText = document.getElementById('urlErrorText');
    const spinnerText = document.getElementById('spinnerText');

    // Tab switching logic
    if (tabUploadBtn && tabUrlBtn) {
        tabUploadBtn.addEventListener('click', () => {
            tabUploadBtn.classList.add('active');
            tabUrlBtn.classList.remove('active');
            uploadTabContent.style.display = 'block';
            urlTabContent.style.display = 'none';
        });

        tabUrlBtn.addEventListener('click', () => {
            tabUrlBtn.classList.add('active');
            tabUploadBtn.classList.remove('active');
            urlTabContent.style.display = 'block';
            uploadTabContent.style.display = 'none';
        });
    }

    // URL Fetching handler
    if (fetchUrlBtn) {
        fetchUrlBtn.addEventListener('click', handleUrlFetch);
    }
    if (urlInput) {
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleUrlFetch();
        });
    }

    async function handleUrlFetch() {
        const urlVal = urlInput.value.trim();
        if (!urlVal) {
            showUrlError('Please enter a public video URL.');
            return;
        }

        urlErrorText.style.display = 'none';
        uploadSpinner.style.display = 'block';
        if (spinnerText) spinnerText.innerText = 'Fetching and downloading video from URL via yt-dlp...';
        fetchUrlBtn.disabled = true;

        try {
            const res = await fetch('/api/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: urlVal })
            });

            const rawText = await res.text();
            let data = {};
            try {
                data = JSON.parse(rawText);
            } catch (_) {
                data = { error: `Server returned status ${res.status}: ${rawText.slice(0, 200)}` };
            }
            fetchUrlBtn.disabled = false;
            uploadSpinner.style.display = 'none';

            if (!res.ok || data.success === false || data.error) {
                showUrlError(data.error || 'Failed to download video from URL.');
                return;
            }

            data.video_path = data.local_path;
            uploadedVideoData = data;
            displayMetadataCard(data);

        } catch (err) {
            fetchUrlBtn.disabled = false;
            uploadSpinner.style.display = 'none';
            showUrlError(`Network error fetching URL: ${err.message}`);
        }
    }

    function showUrlError(msg) {
        if (urlErrorText) {
            urlErrorText.innerText = `⚠️ ${msg}`;
            urlErrorText.style.display = 'block';
        } else {
            alert(msg);
        }
    }

    // Prevent browser from navigating away if file is dropped outside dropzone
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    // Drag and Drop handlers
    uploadZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.stopPropagation();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadVideoFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    uploadZone.addEventListener('click', (e) => {
        if (e.target === fileInput) return;
        fileInput.value = '';
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const selectedFile = fileInput.files[0];
            fileInput.value = '';
            uploadVideoFile(selectedFile);
        }
    });

    async function uploadVideoFile(file) {
        const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
        const dotIdx = file.name.lastIndexOf('.');
        const ext = dotIdx >= 0 ? file.name.substring(dotIdx).toLowerCase() : '';
        
        if (!allowed.includes(ext)) {
            alert(`Unsupported file format. Allowed: ${allowed.join(', ')}`);
            return;
        }

        uploadSpinner.style.display = 'block';
        if (spinnerText) spinnerText.innerText = 'Validating video file and reading metadata...';
        uploadZone.style.opacity = '0.5';
        uploadZone.style.pointerEvents = 'none';

        const formData = new FormData();
        formData.append('video', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const rawText = await res.text();
            let data = {};
            try {
                data = JSON.parse(rawText);
            } catch (_) {
                data = { error: `Server returned status ${res.status}: ${rawText.slice(0, 200)}` };
            }

            uploadSpinner.style.display = 'none';
            uploadZone.style.opacity = '1';
            uploadZone.style.pointerEvents = 'auto';

            if (!res.ok || data.error) {
                alert(`Upload Error: ${data.error || 'Failed to parse video metadata'}`);
                return;
            }

            data.source_type = 'upload';
            uploadedVideoData = data;
            displayMetadataCard(data);

        } catch (err) {
            uploadSpinner.style.display = 'none';
            uploadZone.style.opacity = '1';
            uploadZone.style.pointerEvents = 'auto';
            alert(`Network error uploading video: ${err.message}`);
        }
    }

    function displayMetadataCard(data) {
        uploadFormCard.style.display = 'none';
        configCard.style.display = 'block';

        const meta = data.metadata;
        metaFilename.innerText = data.filename;
        metaDuration.innerText = `${meta.duration_seconds}s`;
        metaResolution.innerText = `${meta.resolution || meta.width + 'x' + meta.height} (${meta.fps || 30} FPS)`;
        if (metaAudio) metaAudio.innerText = meta.has_audio ? 'Available' : 'None';

        videoPreviewPlayer.src = data.video_url;

        // Auto duration setup
        const origSec = meta.duration_seconds;
        durationChips.forEach(chip => {
            const chipSec = chip.dataset.seconds;
            if (chipSec !== 'custom' && parseFloat(chipSec) >= origSec) {
                chip.classList.add('disabled');
                chip.classList.remove('active');
                chip.style.opacity = '0.4';
                chip.style.pointerEvents = 'none';
            } else {
                chip.classList.remove('disabled');
                chip.style.opacity = '1';
                chip.style.pointerEvents = 'auto';
            }
        });

        let defaultChip = document.querySelector('.duration-chip[data-seconds="120"]:not(.disabled)') || 
                          document.querySelector('.duration-chip[data-seconds="60"]:not(.disabled)');
        if (defaultChip) {
            durationChips.forEach(c => c.classList.remove('active'));
            defaultChip.classList.add('active');
            if (customDurationBox) customDurationBox.style.display = 'none';
            selectedDurationSec = parseFloat(defaultChip.dataset.seconds);
        } else {
            selectedDurationSec = Math.max(5, Math.floor(origSec * 0.3));
            const customChip = document.querySelector('.duration-chip[data-seconds="custom"]');
            durationChips.forEach(c => c.classList.remove('active'));
            if (customChip) customChip.classList.add('active');
            if (customDurationBox) customDurationBox.style.display = 'block';
            if (customDurationInput) customDurationInput.value = String(selectedDurationSec);
        }
    }

    function parseCustomDurationSeconds(rawVal) {
        if (!rawVal) return null;
        const trimmed = rawVal.trim();
        if (!trimmed) return null;
        if (/^\d+(\.\d+)?$/.test(trimmed)) {
            const sec = parseFloat(trimmed);
            return sec > 0 ? sec : null;
        }
        const parts = trimmed.split(':').map(p => p.trim());
        if (parts.length === 2 && parts.every(p => /^\d+$/.test(p))) {
            const total = parseInt(parts[0], 10) * 60 + parseInt(parts[1], 10);
            return total > 0 ? total : null;
        }
        if (parts.length === 3 && parts.every(p => /^\d+$/.test(p))) {
            const total = parseInt(parts[0], 10) * 3600 + parseInt(parts[1], 10) * 60 + parseInt(parts[2], 10);
            return total > 0 ? total : null;
        }
        return null;
    }

    // Duration chip selection
    durationChips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (chip.classList.contains('disabled')) return;
            durationChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            const val = chip.dataset.seconds;
            if (val === 'custom') {
                customDurationBox.style.display = 'block';
                const parsed = parseCustomDurationSeconds(customDurationInput.value);
                if (parsed) selectedDurationSec = parsed;
            } else {
                customDurationBox.style.display = 'none';
                selectedDurationSec = parseFloat(val);
                startSummarizeBtn.disabled = false;
            }
        });
    });

    if (customDurationInput) {
        customDurationInput.addEventListener('input', () => {
            const parsed = parseCustomDurationSeconds(customDurationInput.value);
            if (parsed) {
                selectedDurationSec = parsed;
            }
        });
    }

    // Language chip selection
    langChips.forEach(chip => {
        chip.addEventListener('click', () => {
            langChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            selectedLanguage = chip.dataset.lang;
        });
    });

    // Start Summarization Button Click
    startSummarizeBtn.addEventListener('click', async () => {
        const activeDurChip = document.querySelector('.duration-chip.active');
        if (activeDurChip && activeDurChip.dataset.seconds === 'custom') {
            const parsed = parseCustomDurationSeconds(customDurationInput ? customDurationInput.value : '');
            if (!parsed) {
                alert('Please enter a valid custom duration (e.g. 03:30 or 210).');
                return;
            }
            selectedDurationSec = parsed;
        }
        if (!uploadedVideoData || !selectedDurationSec) return;

        const summaryTypeRadio = document.querySelector('input[name="summaryType"]:checked');
        const captionOptionRadio = document.querySelector('input[name="captionOption"]:checked');

        const summaryType = summaryTypeRadio ? summaryTypeRadio.value : 'hybrid';
        const captionOption = captionOptionRadio ? captionOptionRadio.value : 'original';

        configCard.style.display = 'none';
        processingCard.style.display = 'block';

        try {
            statusText.innerText = 'Initializing AI lecture processing pipeline...';
            progressFill.style.width = '5%';

            const reqBody = {
                video_path: uploadedVideoData.video_path,
                target_duration_sec: selectedDurationSec,
                summary_type: summaryType,
                language: selectedLanguage,
                caption_option: captionOption
            };

            const res = await fetch('/api/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });

            const data = await res.json();

            if (!res.ok || data.error) {
                alert(`Error starting summarization: ${data.error || 'Unknown error'}`);
                location.reload();
                return;
            }

            startElapsedTimer();
            listenToSSEProgress(data.job_id);

        } catch (err) {
            alert(`Network error starting summarization: ${err.message}`);
            location.reload();
        }
    });

    function startElapsedTimer() {
        startTimeSec = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTimeSec) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            const timerStr = `Elapsed: ${m}m ${s}s`;
            const timerEl = document.getElementById('elapsedTimerText');
            if (timerEl) {
                timerEl.innerText = timerStr;
            }
        }, 1000);
    }

    // Real-Time Server-Sent Events (SSE) Listener (Requirement 10 & 11)
    function listenToSSEProgress(jobId) {
        const eventSource = new EventSource(`/api/stream/${jobId}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const stage = data.stage;
                const progress = data.progress || 0;
                const message = data.message || 'Processing...';

                // Update Progress Fill and Percentages
                progressFill.style.width = `${progress}%`;
                if (progressPctText) progressPctText.innerText = `${progress}%`;
                if (statusText) statusText.innerText = message;

                // Update Step Checklist
                updateStepChecklist(stage);

                if (stage === 'COMPLETED') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    progressFill.style.width = '100%';
                    if (progressPctText) progressPctText.innerText = '100%';
                    if (statusText) statusText.innerText = 'Lecture summarization completed! Redirecting to result dashboard...';
                    setTimeout(() => {
                        window.location.href = `/result/${jobId}`;
                    }, 500);
                } else if (stage === 'FAILED') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    alert(`Summarization Error: ${data.error || message}`);
                    location.reload();
                }
            } catch (err) {
                console.error('Error parsing SSE event:', err);
            }
        };

        eventSource.onerror = (err) => {
            console.warn('SSE stream reconnecting/fallback polling...', err);
        };
    }

    function updateStepChecklist(stage) {
        const stageOrder = [
            'VALIDATING', 'EXTRACTING_AUDIO', 'PREPROCESSING_AUDIO', 'TRANSCRIBING',
            'POST_PROCESSING_TRANSCRIPT', 'ANALYZING_TRANSCRIPT', 'SCORING_IMPORTANCE',
            'GENERATING_SUMMARY', 'ALIGNING_SUMMARY', 'SELECTING_SEGMENTS', 'TRANSLATING',
            'EXTRACTING_CLIPS', 'JOINING_CLIPS', 'PREPARING_SUBTITLES', 'FINALIZING', 'COMPLETED'
        ];

        const currentIdx = stageOrder.indexOf(stage);
        const stepItems = document.querySelectorAll('.step-item');

        stepItems.forEach((item, idx) => {
            const itemStage = item.dataset.stage;
            const itemIdx = stageOrder.indexOf(itemStage);

            if (currentIdx > itemIdx || stage === 'COMPLETED') {
                item.className = 'step-item completed';
                item.querySelector('.step-status-icon').innerText = '✓';
            } else if (currentIdx === itemIdx) {
                item.className = 'step-item active';
                item.querySelector('.step-status-icon').innerText = '➜';
            } else {
                item.className = 'step-item';
                item.querySelector('.step-status-icon').innerText = idx + 1;
            }
        });
    }
});
