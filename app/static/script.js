let selectedFiles = [];

// DOM elementlari
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');

// Rasm tanlanganda
imageInput.addEventListener('change', function(e) {
    const files = Array.from(e.target.files);

    if (files.length > 0) {
        selectedFiles = files;
        displayImagePreviews(files);
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';

    const files = Array.from(e.dataTransfer.files).filter(file =>
        file.type === 'image/png' || file.type === 'image/jpeg' || file.type === 'image/jpg'
    );

    if (files.length > 0) {
        selectedFiles = files;
        displayImagePreviews(files);
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    }
});

// Rasmlarni ko'rsatish
function displayImagePreviews(files) {
    imagePreview.innerHTML = '';

    files.forEach(file => {
        const reader = new FileReader();

        reader.onload = function(e) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';

            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <div class="filename">${file.name}</div>
            `;

            imagePreview.appendChild(previewItem);
        };

        reader.readAsDataURL(file);
    });
}

// Tekshirish tugmasi
predictBtn.addEventListener('click', async function() {
    if (selectedFiles.length === 0) {
        alert('Iltimos, avval rasm tanlang!');
        return;
    }

    // Loading ko'rsatish
    loading.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        let results;

        if (selectedFiles.length === 1) {
            // Bitta rasm uchun
            results = await predictSingle(selectedFiles[0]);
        } else {
            // Ko'p rasmlar uchun
            results = await predictBatch(selectedFiles);
        }

        displayResults(results);
    } catch (error) {
        alert('Xatolik yuz berdi: ' + error.message);
        console.error('Error:', error);
    } finally {
        loading.style.display = 'none';
    }
});

// Bitta rasmni bashorat qilish
async function predictSingle(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Server xatosi');
    }

    const data = await response.json();
    return [data.data];
}

// Ko'p rasmlarni bashorat qilish
async function predictBatch(files) {
    const formData = new FormData();

    files.forEach(file => {
        formData.append('files', file);
    });

    const response = await fetch('/predict-batch', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Server xatosi');
    }

    const data = await response.json();
    return data.results;
}

// Natijalarni ko'rsatish
function displayResults(results) {
    resultsContainer.innerHTML = '';
    resultsSection.style.display = 'block';

    results.forEach((result, index) => {
        const resultCard = createResultCard(result, index);
        resultsContainer.appendChild(resultCard);
    });

    // Natijalar bo'limiga scroll qilish
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Natija kartasini yaratish
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';

    // Ishonch darajasi rangini aniqlash
    const confidence = result.confidence;
    let confidenceColor = '#4caf50'; // Yashil

    if (confidence < 0.5) {
        confidenceColor = '#f44336'; // Qizil
    } else if (confidence < 0.7) {
        confidenceColor = '#ff9800'; // Orange
    }

    // Ehtimolliklar bo'yicha progressbar yaratish
    let probabilitiesHTML = '<div class="probabilities"><h4>Barcha ehtimolliklar:</h4>';

    for (const [className, probability] of Object.entries(result.all_probabilities)) {
        const percentage = (probability * 100).toFixed(1);
        probabilitiesHTML += `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${className}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }
    probabilitiesHTML += '</div>';

    card.innerHTML = `
        <div class="result-info">
            <div class="result-class">${result.predicted_class}</div>
            <div class="result-confidence" style="color: ${confidenceColor}">
                Ishonch: ${(confidence * 100).toFixed(1)}%
            </div>
            ${result.filename ? `<div style="color: #666; font-size: 0.9rem; margin-top: 5px;">Fayl: ${result.filename}</div>` : ''}
        </div>
        ${probabilitiesHTML}
    `;

    return card;
}

// Tozalash tugmasi
clearBtn.addEventListener('click', function() {
    selectedFiles = [];
    imagePreview.innerHTML = '';
    imageInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
});

// Sahifa yuklanganda API health check
window.addEventListener('load', async function() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('API Status:', data);

        if (!data.model_loaded) {
            console.warn('Model yuklanmagan! Iltimos, modelni o\'rgating.');
        }
    } catch (error) {
        console.error('API bilan bog\'lanishda xatolik:', error);
    }
});
