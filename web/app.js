// Multimodal Emotion Analysis - Frontend JavaScript

const API_BASE = '/api';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
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
});

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
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
            handleAudioFile(audioFile);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
        recordingStatus.classList.remove('hidden');
        
        // Setup audio visualization
        setupRecordingVisualization(stream);
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error accessing microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Start Recording';
        recordingStatus.classList.add('hidden');
        
        // Stop visualization
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        if (audioContext) {
            audioContext.close();
        }
    }
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
    
    // Audio Emotion (only one emotion display now)
    displayEmotions(data.audio_emotion, audioEmotionDisplay);
    
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

