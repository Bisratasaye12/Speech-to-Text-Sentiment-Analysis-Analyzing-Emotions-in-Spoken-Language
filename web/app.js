// Multimodal Emotion Analysis - Frontend JavaScript

const API_BASE = '/api';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const micPermissionNote = document.getElementById('micPermissionNote');
const waveformContainer = document.getElementById('waveformContainer');
const waveformCanvas = document.getElementById('waveformCanvas');
const audioPlayer = document.getElementById('audioPlayer');
const audioElement = document.getElementById('audioElement');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const transcriptionText = document.getElementById('transcriptionText');
const audioEmotionDisplay = document.getElementById('audioEmotionDisplay');

// State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioContext = null;
let analyser = null;
let animationFrame = null;

// Emotion emojis
const EMOTION_EMOJIS = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupFileUpload();
    setupRecording();
    setupWaveform();
    checkMicrophonePermission();
});

// Check microphone permission on page load (silent check, only show warnings)
async function checkMicrophonePermission() {
    try {
        // Check if HTTPS or localhost (required for getUserMedia)
        const isSecure = window.location.protocol === 'https:' || 
                        window.location.hostname === 'localhost' || 
                        window.location.hostname === '127.0.0.1';
        
        if (!isSecure) {
            micPermissionNote.classList.remove('hidden');
            micPermissionNote.innerHTML = '<small>⚠️ Microphone access requires HTTPS. Please use HTTPS or localhost.</small>';
            micPermissionNote.style.color = 'var(--warning)';
        }
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn('Microphone access not supported in this browser');
            recordBtn.disabled = true;
            recordBtn.title = 'Microphone access not supported. Please use a modern browser.';
            recordBtn.style.opacity = '0.5';
            recordBtn.style.cursor = 'not-allowed';
            micPermissionNote.classList.remove('hidden');
            micPermissionNote.innerHTML = '<small>⚠️ Microphone access not supported in this browser.</small>';
            micPermissionNote.style.color = 'var(--error)';
            return;
        }
        
        // Only show permission note if there's an actual problem
        // Don't show it for normal "prompt" state - that's expected behavior
        if (navigator.permissions && navigator.permissions.query) {
            try {
                const permission = await navigator.permissions.query({ name: 'microphone' });
                
                if (permission.state === 'denied') {
                    // Only show note if permission is denied
                    recordBtn.disabled = true;
                    recordBtn.title = 'Microphone access denied. Please enable it in browser settings.';
                    recordBtn.style.opacity = '0.5';
                    recordBtn.style.cursor = 'not-allowed';
                    micPermissionNote.classList.remove('hidden');
                    micPermissionNote.innerHTML = '<small>❌ Microphone access denied. Please enable it in browser settings.</small>';
                    micPermissionNote.style.color = 'var(--error)';
                } else {
                    // Hide note for "prompt" or "granted" - these are normal states
                    micPermissionNote.classList.add('hidden');
                }
                
                // Listen for permission changes
                permission.onchange = () => {
                    if (permission.state === 'granted') {
                        recordBtn.disabled = false;
                        recordBtn.title = 'Click to start recording';
                        recordBtn.style.opacity = '1';
                        recordBtn.style.cursor = 'pointer';
                        micPermissionNote.classList.add('hidden');
                    } else if (permission.state === 'denied') {
                        recordBtn.disabled = true;
                        recordBtn.title = 'Microphone access denied. Please enable it in browser settings.';
                        recordBtn.style.opacity = '0.5';
                        recordBtn.style.cursor = 'not-allowed';
                        micPermissionNote.classList.remove('hidden');
                        micPermissionNote.innerHTML = '<small>❌ Microphone access denied. Please enable it in browser settings.</small>';
                        micPermissionNote.style.color = 'var(--error)';
                    } else {
                        micPermissionNote.classList.add('hidden');
                    }
                };
            } catch (permError) {
                // Permissions API might not support 'microphone' name in some browsers
                // This is fine - just hide the note and let the normal prompt work
                micPermissionNote.classList.add('hidden');
            }
        } else {
            // Permissions API not available - hide note, normal prompt will work
            micPermissionNote.classList.add('hidden');
        }
    } catch (error) {
        // Hide note on any error - normal prompt will work
        console.log('Could not check microphone permission:', error);
        micPermissionNote.classList.add('hidden');
    }
}

// File Upload Setup
function setupFileUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File selected
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleAudioFile(file);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            handleAudioFile(file);
        }
    });
}

// Recording Setup
function setupRecording() {
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });
}

async function startRecording() {
    try {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Microphone access is not supported in this browser. Please use a modern browser like Chrome, Firefox, or Safari.');
        }

        // Request microphone access with better error handling
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000  // Match backend expected sample rate
            } 
        });
        
        // Determine the best MIME type for MediaRecorder
        const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/wav'
        ];
        
        let selectedMimeType = 'audio/webm';
        for (const mimeType of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mimeType)) {
                selectedMimeType = mimeType;
                break;
            }
        }
        
        // Create MediaRecorder with the best supported format
        const options = { mimeType: selectedMimeType };
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            stopRecording();
            alert('Error during recording. Please try again.');
        };

        mediaRecorder.onstop = async () => {
            try {
                // Create blob from recorded chunks
                const audioBlob = new Blob(audioChunks, { type: selectedMimeType });
                
                // Convert to WAV format for backend compatibility
                const audioFile = await convertToWav(audioBlob, selectedMimeType);
                handleAudioFile(audioFile);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            } catch (error) {
                console.error('Error processing recording:', error);
                alert('Error processing recording. Please try again.');
            }
        };

        // Start recording with timeslice for better chunk handling
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;
        
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
        recordingStatus.classList.remove('hidden');
        
        // Setup audio visualization
        setupRecordingVisualization(stream);
        
    } catch (error) {
        console.error('Error starting recording:', error);
        
        let errorMessage = 'Error accessing microphone. ';
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage += 'Please allow microphone access in your browser settings and try again.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMessage += 'No microphone found. Please connect a microphone and try again.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMessage += 'Microphone is already in use by another application.';
        } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
            errorMessage += 'Microphone does not support required settings.';
        } else {
            errorMessage += error.message || 'Please check your browser permissions.';
        }
        
        alert(errorMessage);
        isRecording = false;
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Record';
        recordingStatus.classList.add('hidden');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        try {
            // Stop recording
            if (mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        } catch (error) {
            console.error('Error stopping recording:', error);
        }
        
        isRecording = false;
        
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Record';
        recordingStatus.classList.add('hidden');
        
        // Stop visualization
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
        if (audioContext) {
            audioContext.close().catch(console.error);
            audioContext = null;
        }
        if (analyser) {
            analyser = null;
        }
    }
}

// Convert audio blob to WAV format for backend compatibility
async function convertToWav(audioBlob, mimeType) {
    try {
        // If already WAV, return as-is
        if (mimeType.includes('wav')) {
            return new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
        }
        
        // Decode audio data
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Convert to WAV
        const wavBlob = audioBufferToWav(audioBuffer);
        audioContext.close();
        
        return new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
    } catch (error) {
        console.warn('Could not convert to WAV, sending original format:', error);
        // Fallback: send original format
        const extension = mimeType.includes('webm') ? 'webm' : 
                         mimeType.includes('ogg') ? 'ogg' : 
                         mimeType.includes('mp4') ? 'm4a' : 'wav';
        return new File([audioBlob], `recording.${extension}`, { type: mimeType });
    }
}

// Convert AudioBuffer to WAV Blob
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const length = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * numChannels * bytesPerSample);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * numChannels * bytesPerSample, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length * numChannels * bytesPerSample, true);
    
    // Convert float samples to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

function setupRecordingVisualization(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    waveformContainer.classList.remove('hidden');
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    function draw() {
        if (!isRecording) return;
        
        animationFrame = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        
        ctx.fillStyle = '#f5f5f7';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#007aff';
        ctx.beginPath();
        
        const sliceWidth = canvas.width / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    
    draw();
}

// Waveform Setup
function setupWaveform() {
    // Canvas will be used for both recording and playback visualization
}

function drawWaveform(audioBuffer) {
    waveformContainer.classList.remove('hidden');
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / canvas.width);
    const amp = canvas.height / 2;
    
    ctx.fillStyle = '#f5f5f7';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#007aff';
    ctx.beginPath();
    
    for (let i = 0; i < canvas.width; i++) {
        const min = Math.min(...data.slice(i * step, (i + 1) * step));
        const max = Math.max(...data.slice(i * step, (i + 1) * step));
        
        const x = i;
        const y1 = amp + min * amp;
        const y2 = amp + max * amp;
        
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
    }
    
    ctx.stroke();
}

// Handle Audio File
async function handleAudioFile(file) {
    // Show audio player
    const audioUrl = URL.createObjectURL(file);
    audioElement.src = audioUrl;
    audioPlayer.classList.remove('hidden');
    
    // Draw waveform if possible
    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        drawWaveform(audioBuffer);
    } catch (error) {
        console.warn('Could not draw waveform:', error);
    }
    
    // Analyze audio
    await analyzeAudio(file);
}

// Analyze Audio
async function analyzeAudio(file) {
    // Show processing
    resultsSection.classList.add('hidden');
    processingSection.classList.remove('hidden');
    
    try {
        const formData = new FormData();
        formData.append('audio', file);
        
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        alert(`Error analyzing audio: ${error.message}`);
    } finally {
        processingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
    }
}

// Display Results
function displayResults(data) {
    // Transcription
    transcriptionText.textContent = data.transcription || 'No transcription available';
    
    // Use fused_emotion (multimodal fusion) as primary display, fallback to audio_emotion
    const primaryEmotion = data.fused_emotion || data.audio_emotion;
    displayEmotions(primaryEmotion, audioEmotionDisplay);
    
    // Show results
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayEmotions(emotionData, container) {
    container.innerHTML = '';
    
    // Dominant emotion (large display)
    if (emotionData.dominant) {
        const dominantDiv = document.createElement('div');
        dominantDiv.className = 'dominant-emotion';
        dominantDiv.innerHTML = `
            <div class="dominant-emotion-emoji">${emotionData.dominant.emoji}</div>
            <div class="dominant-emotion-label">${emotionData.dominant.label}</div>
            <div class="dominant-emotion-confidence">${(emotionData.dominant.confidence * 100).toFixed(1)}% confidence</div>
        `;
        container.appendChild(dominantDiv);
    }
    
    // All emotions grid
    const gridDiv = document.createElement('div');
    gridDiv.className = 'emotion-display';
    
    if (emotionData.all) {
        const emotions = Object.entries(emotionData.all);
        emotions.sort((a, b) => b[1].probability - a[1].probability);
        
        emotions.forEach(([label, emotion], index) => {
            const isDominant = emotionData.dominant && label === emotionData.dominant.label;
            const item = document.createElement('div');
            item.className = `emotion-item ${isDominant ? 'dominant' : ''}`;
            item.style.animationDelay = `${index * 0.1}s`;
            
            item.innerHTML = `
                <span class="emotion-emoji">${emotion.emoji}</span>
                <div class="emotion-label">${label}</div>
                <div class="emotion-confidence">${(emotion.probability * 100).toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: ${emotion.probability * 100}%"></div>
                </div>
            `;
            
            gridDiv.appendChild(item);
        });
    }
    
    container.appendChild(gridDiv);
}

