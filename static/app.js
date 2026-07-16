const form = document.querySelector('#analysisForm');
const fileInput = document.querySelector('#files');
const fileList = document.querySelector('#fileList');
const dropzone = document.querySelector('#dropzone');
const analyzeButton = document.querySelector('#analyzeButton');
const modelStatus = document.querySelector('#modelStatus');
const emptyState = document.querySelector('#emptyState');
const loadingState = document.querySelector('#loadingState');
const results = document.querySelector('#results');
const resultNavigator = document.querySelector('#resultNavigator');
const message = document.querySelector('#message');
const resultTemplate = document.querySelector('#resultTemplate');

let selectedFiles = [];
let analysisFiles = [];
let analysisResults = [];
let activeResultIndex = 0;
// === ПЕРЕКЛЮЧЕНИЕ ТЕМ ===
const themeButtons = document.querySelectorAll('.theme-button');
const savedTheme = localStorage.getItem('geovision-theme') || 'light';

// Применяем сохранённую тему при загрузке
document.documentElement.setAttribute('data-theme', savedTheme);
themeButtons.forEach(button => {
    button.classList.toggle('active', button.dataset.theme === savedTheme);
});

// Обработчики кликов по кнопкам тем
themeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const theme = button.dataset.theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('geovision-theme', theme);
        
        themeButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    });
});
const formatNumber = (value, digits = 0) => new Intl.NumberFormat('ru-RU', {
    maximumFractionDigits: digits,
}).format(Number(value || 0));

const formatMoney = (value) => {
    const number = Number(value || 0);
    if (number >= 1_000_000_000) return `$${formatNumber(number / 1_000_000_000, 2)} млрд`;
    if (number >= 1_000_000) return `$${formatNumber(number / 1_000_000, 2)} млн`;
    return `$${formatNumber(number, 0)}`;
};

function setMessage(text = '', type = '') {
    message.hidden = !text;
    message.textContent = text;
    message.className = `message ${type}`.trim();
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        modelStatus.className = `status status-${status.mode === 'model' ? 'model' : 'demo'}`;
        modelStatus.innerHTML = `<span class="status-dot"></span><span>${status.model_loaded ? `${status.architecture} · ${status.device}` : 'Демонстрационный режим'}</span>`;
        if (status.error && !status.model_loaded) modelStatus.title = status.error;
    } catch (error) {
        modelStatus.className = 'status status-error';
        modelStatus.innerHTML = '<span class="status-dot"></span><span>Сервер недоступен</span>';
    }
}

function renderFileList() {
    fileList.replaceChildren();
    selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-pill';
        const name = document.createElement('span');
        name.textContent = `${file.name} · ${formatNumber(file.size / 1024, 0)} КБ`;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.textContent = '×';
        remove.setAttribute('aria-label', `Удалить ${file.name}`);
        remove.addEventListener('click', () => {
            selectedFiles.splice(index, 1);
            renderFileList();
        });
        item.append(name, remove);
        fileList.append(item);
    });
}

function addFiles(files) {
    const incoming = [...files].filter((file) => file.type.startsWith('image/'));
    const known = new Set(selectedFiles.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
    for (const file of incoming) {
        const key = `${file.name}-${file.size}-${file.lastModified}`;
        if (!known.has(key) && selectedFiles.length < 20) {
            selectedFiles.push(file);
            known.add(key);
        }
    }
    renderFileList();
}

function appendFormParameters(payload) {
    for (const element of [...form.elements]) {
        if (!element.name || element.name === 'files' || element.type === 'file') continue;
        if (element.value !== '') payload.append(element.name, element.value);
    }
}

fileInput.addEventListener('change', () => addFiles(fileInput.files));

for (const eventName of ['dragenter', 'dragover']) {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add('dragging');
    });
}

for (const eventName of ['dragleave', 'drop']) {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove('dragging');
    });
}

dropzone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));

function loadImage(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error('Не удалось подготовить изображение для редактора'));
        image.src = source;
    });
}

function renderNavigator() {
    resultNavigator.replaceChildren();
    resultNavigator.hidden = analysisResults.length <= 1;
    analysisResults.forEach((result, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.classList.toggle('active', index === activeResultIndex);
        button.textContent = `${String(index + 1).padStart(2, '0')} ${result.filename}`;
        button.title = result.filename;
        button.addEventListener('click', () => showResult(index));
        resultNavigator.append(button);
    });
}

function renderWellTable(card, scenario) {
    const tbody = card.querySelector('.well-table-body');
    const noWells = card.querySelector('.no-wells');
    const wells = scenario.wells || [];
    tbody.replaceChildren();
    noWells.hidden = wells.length > 0;
    card.querySelector('.well-count').textContent = `${wells.length} скв.`;
    for (const well of wells) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${well.rank}</td>
            <td>${formatNumber(well.x_m, 2)}</td>
            <td>${formatNumber(well.y_m, 2)}</td>
            <td>${formatNumber(well.mean_probability * 100, 1)}%</td>
            <td>${formatNumber(well.marginal_area_m2 / 1_000_000, 3)} км²</td>
            <td>${formatNumber(well.marginal_volume_m3, 0)}</td>
            <td><span class="score-pill">${formatNumber(well.score, 3)}</span></td>
        `;
        tbody.append(row);
    }
}

async function setupMaskEditor(card, result, resultIndex) {
    const canvas = card.querySelector('.editor-canvas');
    const originalImage = await loadImage(result.images.original);
    const binaryMaskImage = await loadImage(result.images.mask_binary);

    canvas.width = originalImage.naturalWidth;
    canvas.height = originalImage.naturalHeight;

    const displayContext = canvas.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskContext = maskCanvas.getContext('2d', { willReadFrequently: true });
    maskContext.drawImage(binaryMaskImage, 0, 0, canvas.width, canvas.height);

    const loadedMask = maskContext.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 0; index < loadedMask.data.length; index += 4) {
        const value = loadedMask.data[index];
        loadedMask.data[index] = 255;
        loadedMask.data[index + 1] = 255;
        loadedMask.data[index + 2] = 255;
        loadedMask.data[index + 3] = value;
    }
    maskContext.clearRect(0, 0, canvas.width, canvas.height);
    maskContext.putImageData(loadedMask, 0, 0);

    const initialMask = maskContext.getImageData(0, 0, canvas.width, canvas.height);
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    const overlayContext = overlayCanvas.getContext('2d');

    const history = [];
    let tool = 'brush';
    let drawing = false;
    let lastPoint = null;

    const brushSize = card.querySelector('.brush-size');
    const originalOpacity = card.querySelector('.original-opacity');
    const maskOpacity = card.querySelector('.mask-opacity');

    function drawComposite() {
        displayContext.save();
        displayContext.globalAlpha = 1;
        displayContext.fillStyle = '#0c1721';
        displayContext.fillRect(0, 0, canvas.width, canvas.height);
        displayContext.globalAlpha = Number(originalOpacity.value);
        displayContext.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
        displayContext.restore();

        overlayContext.clearRect(0, 0, canvas.width, canvas.height);
        overlayContext.globalCompositeOperation = 'source-over';
        overlayContext.fillStyle = '#165bea';
        overlayContext.fillRect(0, 0, canvas.width, canvas.height);
        overlayContext.globalCompositeOperation = 'destination-in';
        overlayContext.drawImage(maskCanvas, 0, 0);
        overlayContext.globalCompositeOperation = 'source-over';

        displayContext.save();
        displayContext.globalAlpha = Number(maskOpacity.value);
        displayContext.drawImage(overlayCanvas, 0, 0);
        displayContext.restore();
    }

    function pointFromEvent(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * canvas.width / rect.width,
            y: (event.clientY - rect.top) * canvas.height / rect.height,
            scale: canvas.width / rect.width,
        };
    }

    function paintPoint(point) {
        const width = Number(brushSize.value) * point.scale;
        maskContext.save();
        maskContext.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
        maskContext.fillStyle = '#ffffff';
        maskContext.beginPath();
        maskContext.arc(point.x, point.y, width / 2, 0, Math.PI * 2);
        maskContext.fill();
        maskContext.restore();
    }

    function paintLine(from, to) {
        const width = Number(brushSize.value) * to.scale;
        maskContext.save();
        maskContext.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
        maskContext.strokeStyle = '#ffffff';
        maskContext.lineCap = 'round';
        maskContext.lineJoin = 'round';
        maskContext.lineWidth = width;
        maskContext.beginPath();
        maskContext.moveTo(from.x, from.y);
        maskContext.lineTo(to.x, to.y);
        maskContext.stroke();
        maskContext.restore();
    }

    canvas.addEventListener('pointerdown', (event) => {
        history.push(maskContext.getImageData(0, 0, canvas.width, canvas.height));
        if (history.length > 12) history.shift();
        drawing = true;
        lastPoint = pointFromEvent(event);
        paintPoint(lastPoint);
        drawComposite();
        canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener('pointermove', (event) => {
        if (!drawing) return;
        const point = pointFromEvent(event);
        paintLine(lastPoint, point);
        lastPoint = point;
        drawComposite();
    });

    const stopDrawing = (event) => {
        drawing = false;
        lastPoint = null;
        if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };

    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointercancel', stopDrawing);

    card.querySelectorAll('[data-tool]').forEach((button) => {
        button.addEventListener('click', () => {
            card.querySelectorAll('[data-tool]').forEach((candidate) => candidate.classList.remove('active'));
            button.classList.add('active');
            tool = button.dataset.tool;
        });
    });

    originalOpacity.addEventListener('change', drawComposite);
    maskOpacity.addEventListener('change', drawComposite);

    card.querySelector('.undo-mask').addEventListener('click', () => {
        const previous = history.pop();
        if (previous) {
            maskContext.putImageData(previous, 0, 0);
            drawComposite();
        }
    });

    card.querySelector('.reset-mask').addEventListener('click', () => {
        history.push(maskContext.getImageData(0, 0, canvas.width, canvas.height));
        maskContext.putImageData(initialMask, 0, 0);
        drawComposite();
    });

    card.querySelector('.recalculate-button').addEventListener('click', async (event) => {
        const button = event.currentTarget;
        const sourceFile = analysisFiles[resultIndex];
        if (!sourceFile) {
            setMessage('Исходный файл недоступен. Запустите анализ заново.', 'error');
            return;
        }

        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = maskCanvas.width;
        exportCanvas.height = maskCanvas.height;
        const exportContext = exportCanvas.getContext('2d');
        exportContext.fillStyle = '#000000';
        exportContext.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
        exportContext.drawImage(maskCanvas, 0, 0);

        const blob = await new Promise((resolve) => exportCanvas.toBlob(resolve, 'image/png'));
        if (!blob) {
            setMessage('Не удалось подготовить маску.', 'error');
            return;
        }

        const payload = new FormData();
        payload.append('file', sourceFile);
        payload.append('mask', blob, 'edited_mask.png');
        appendFormParameters(payload);

        button.disabled = true;
        button.textContent = 'Пересчёт…';
        setMessage();

        try {
            const response = await fetch('/api/recalculate', { method: 'POST', body: payload });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Не удалось пересчитать результат');
            
            analysisResults[resultIndex] = data.result;
            await showResult(resultIndex);
            setMessage('Коррекция маски применена. Объём и координаты пересчитаны.', 'success');
        } catch (error) {
            setMessage(error.message || 'Ошибка соединения с сервером.', 'error');
            button.disabled = false;
            button.textContent = 'Применить и пересчитать';
        }
    });

    drawComposite();
    return { draw: drawComposite };
}

function buildResultCard(result, index) {
    const fragment = resultTemplate.content.cloneNode(true);
    const card = fragment.querySelector('.result-card');
    
    card.querySelector('.result-index').textContent = String(index + 1).padStart(2, '0');
    card.querySelector('.result-filename').textContent = result.filename;
    card.querySelector('.edit-state').hidden = !result.edited;

    const timing = result.timings_seconds || {};
    const resolution = card.querySelector('.result-resolution');
    resolution.textContent = `${result.width} × ${result.height} px · ${formatNumber(timing.total, 1)} c`;
    resolution.title = `U-Net: ${timing.inference || 0} c; план: ${timing.well_planning || 0} c`;

    const summary = result.summary;
    card.querySelector('[data-metric="area"]').textContent = `${formatNumber(summary.mask_area_km2, 3)} км²`;
    card.querySelector('[data-metric="volume"]').textContent = formatNumber(summary.recoverable_m3, 0);
    card.querySelector('[data-metric="confidence"]').textContent = `${formatNumber(summary.mean_probability * 100, 1)}%`;
    card.querySelector('[data-metric="value"]').textContent = formatMoney(summary.gross_value_usd);

    const image = card.querySelector('.result-image');
    const editorCanvas = card.querySelector('.editor-canvas');
    const editorToolbar = card.querySelector('.editor-toolbar');
    const probabilityScale = card.querySelector('.probability-scale');
    
    // === НОВЫЕ ЭЛЕМЕНТЫ ДЛЯ SWIPE ===
    const swipeContainer = card.querySelector('.swipe-container');
    const swipeWrapper = card.querySelector('.swipe-wrapper');
    const swipeBefore = card.querySelector('.swipe-before');
    const swipeAfter = card.querySelector('.swipe-after');
    const swipeHandle = card.querySelector('.swipe-handle');
    const swipeLayerSelect = card.querySelector('.swipe-layer-select');

    let activeView = 'wells';
    let activeScenario = 'recommended';

    image.src = result.images.wells_recommended || result.images.wells;

    const scenarioButtons = [...card.querySelectorAll('[data-scenario]')];
    for (const button of scenarioButtons) {
        const scenario = result.well_scenarios[button.dataset.scenario];
        button.querySelector('strong').textContent = `${scenario.count} скв.`;
        button.addEventListener('click', () => {
            activeScenario = button.dataset.scenario;
            scenarioButtons.forEach((candidate) => candidate.classList.toggle('active', candidate === button));
            renderWellTable(card, result.well_scenarios[activeScenario]);
            if (activeView === 'wells') image.src = result.images[`wells_${activeScenario}`];
        });
    }

    renderWellTable(card, result.well_scenarios[activeScenario]);

    const editorPromise = setupMaskEditor(card, result, index).catch((error) => {
        setMessage(error.message, 'error');
        return null;
    });

    const tabButtons = [...card.querySelectorAll('[data-view]')];
    tabButtons.forEach((button) => {
        button.addEventListener('click', async () => {
            activeView = button.dataset.view;
            tabButtons.forEach((candidate) => candidate.classList.toggle('active', candidate === button));

            const editing = activeView === 'edit';
            const swiping = activeView === 'swipe';

            // Управляем видимостью всех блоков
            image.hidden = editing || swiping;
            editorCanvas.hidden = !editing;
            editorToolbar.hidden = !editing;
            probabilityScale.hidden = activeView !== 'probability';
            swipeContainer.hidden = !swiping;

            if (editing) {
                const editor = await editorPromise;
                if (editor) editor.draw();
            } else if (swiping) {
                // Инициализация режима Swipe
                swipeBefore.src = result.images.original;
                
                const layerValue = swipeLayerSelect.value;
                const afterKey = layerValue === 'probability' ? 'probability' : 
                                 layerValue === 'mask' ? 'mask' : 
                                 `wells_${activeScenario}`;
                swipeAfter.src = result.images[afterKey];
                
                initSwipeInteraction();
            } else if (activeView === 'wells') {
                image.src = result.images[`wells_${activeScenario}`];
            } else {
                image.src = result.images[activeView];
            }
        });
    });

    // === ЛОГИКА ПЕРЕКЛЮЧЕНИЯ СЛОЕВ В SWIPE ===
    swipeLayerSelect.addEventListener('change', () => {
        if (activeView === 'swipe') {
            const layerValue = swipeLayerSelect.value;
            const afterKey = layerValue === 'probability' ? 'probability' : 
                             layerValue === 'mask' ? 'mask' : 
                             `wells_${activeScenario}`;
            swipeAfter.src = result.images[afterKey];
        }
    });

    // === ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ ПОЛЗУНКА ===
    function initSwipeInteraction() {
        swipeAfter.style.clipPath = 'inset(0 50% 0 0)';
        swipeHandle.style.left = '50%';

        let isDragging = false;

        function updatePosition(clientX) {
            const rect = swipeWrapper.getBoundingClientRect();
            let pos = (clientX - rect.left) / rect.width;
            pos = Math.max(0, Math.min(1, pos));
            const percent = pos * 100;

            swipeAfter.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
            swipeHandle.style.left = `${percent}%`;
        }

        swipeWrapper.onpointerdown = (e) => {
            isDragging = true;
            updatePosition(e.clientX);
            swipeWrapper.setPointerCapture(e.pointerId);
        };
        
        swipeWrapper.onpointermove = (e) => {
            if (isDragging) updatePosition(e.clientX);
        };
        
        swipeWrapper.onpointerup = () => {
            isDragging = false;
        };
        
        swipeWrapper.onpointercancel = () => {
            isDragging = false;
        };
    }

    return card;
}

async function showResult(index) {
    activeResultIndex = index;
    renderNavigator();
    results.replaceChildren();
    const result = analysisResults[index];
    if (!result) return;
    results.append(buildResultCard(result, index));
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    setMessage();
    if (!selectedFiles.length) {
        setMessage('Добавьте хотя бы одно изображение.', 'error');
        return;
    }

    const payload = new FormData();
    selectedFiles.forEach((file) => payload.append('files', file));
    appendFormParameters(payload);

    analyzeButton.disabled = true;
    analyzeButton.querySelector('span:first-child').textContent = 'Выполняется анализ…';
    emptyState.hidden = true;
    resultNavigator.hidden = true;
    results.replaceChildren();
    loadingState.hidden = false;

    try {
        const response = await fetch('/api/analyze', { method: 'POST', body: payload });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Не удалось выполнить анализ');
        
        if (data.warning) setMessage(data.warning);
        
        analysisFiles = [...selectedFiles];
        analysisResults = data.results;
        await showResult(0);
    } catch (error) {
        emptyState.hidden = false;
        setMessage(error.message || 'Ошибка соединения с сервером.', 'error');
    } finally {
        loadingState.hidden = true;
        analyzeButton.disabled = false;
        analyzeButton.querySelector('span:first-child').textContent = 'Выполнить интерпретацию';
    }
});
// === ПЕРЕКЛЮЧЕНИЕ ТЕМ ===
const themeButtons = document.querySelectorAll('.theme-button');
const savedTheme = localStorage.getItem('geovision-theme') || 'light';

// Применяем сохранённую тему при загрузке
document.documentElement.setAttribute('data-theme', savedTheme);
themeButtons.forEach(button => {
    button.classList.toggle('active', button.dataset.theme === savedTheme);
});

// Обработчики кликов по кнопкам тем
themeButtons.forEach(button => {
    button.addEventListener('click', () => {
        const theme = button.dataset.theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('geovision-theme', theme);
        
        themeButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    });
});
loadStatus();