/**
 * MedPredict AI v3 — Frontend Logic
 * Handles:
 *  - Dynamic clinical range coloring as user types
 *  - Form submission & API call
 *  - Result card rendering with animated probability bars
 *  - PDF report download
 */

const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://127.0.0.1:8000'   // local development
    : '';                        // Vercel production: use relative /api/... path

let lastPrediction = null;

// ── Fields sent as integers ───────────────────────────────────────────────────
const INTEGER_FIELDS = new Set([
    'Smoking', 'Physical_Activity', 'Alcohol_Use',
    'Family_History_Diabetes', 'Family_History_Heart'
]);

// ── Disease display icons ─────────────────────────────────────────────────────
const DISEASE_ICONS = {
    Diabetes:      '🩺',
    Heart_Disease: '❤️',
    Liver_Disease: '🫀',
};

// ─────────────────────────────────────────────────────────────────────────────
// DYNAMIC CLINICAL RANGE COLORING
// ─────────────────────────────────────────────────────────────────────────────
function applyRangeColor(input) {
    const raw = parseFloat(input.value);
    input.classList.remove('val-normal', 'val-medium', 'val-high');
    if (isNaN(raw)) return;

    const rangesStr = input.dataset.ranges;
    if (!rangesStr) return;

    const ranges  = JSON.parse(rangesStr);
    const inverse = input.dataset.inverse === 'true';
    const [nLo, nHi] = ranges.normal;
    const [mLo, mHi] = ranges.medium;
    const [hLo, hHi] = ranges.high;

    let cls = '';
    if (raw >= nLo && raw <= nHi)      cls = 'val-normal';
    else if (raw >= mLo && raw <= mHi) cls = 'val-medium';
    else if (raw >= hLo && raw <= hHi) cls = 'val-high';
    else {
        cls = (!inverse)
            ? (raw < nLo ? 'val-normal' : 'val-high')
            : (raw > nLo ? 'val-normal' : 'val-high');
    }
    input.classList.add(cls);
}

// Attach live range coloring
document.querySelectorAll('input[data-ranges]').forEach(input => {
    input.addEventListener('input',  () => applyRangeColor(input));
    input.addEventListener('change', () => applyRangeColor(input));
});

// ─────────────────────────────────────────────────────────────────────────────
// FORM SUBMISSION
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form             = e.target;
    const submitBtn        = document.getElementById('submit-btn');
    const btnText          = submitBtn.querySelector('.btn-text');
    const spinner          = submitBtn.querySelector('.spinner');
    const resultsContainer = document.getElementById('results-container');

    // Loading state
    btnText.textContent = 'Analyzing…';
    submitBtn.querySelector('svg').style.display = 'none';
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;

    // Build data payload
    const formData = new FormData(form);
    const data = {};
    for (const [key, value] of formData.entries()) {
        if (key === 'Gender') {
            data[key] = value;
        } else if (INTEGER_FIELDS.has(key)) {
            data[key] = parseInt(value, 10);
        } else {
            data[key] = parseFloat(value);
        }
    }

    // Validate — all required fields must be present
    const missing = Object.entries(data).filter(([k, v]) =>
        v === '' || v === null || (k !== 'Gender' && isNaN(v))
    );
    if (missing.length > 0 || !data.Gender) {
        showValidationError(resultsContainer);
        resetButton(btnText, spinner, submitBtn);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const err    = await response.json().catch(() => ({}));
            const detail = typeof err.detail === 'string'
                ? err.detail
                : JSON.stringify(err.detail || 'Unknown server error');
            throw new Error(detail || `Server error ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            lastPrediction = { data, predictions: result.predictions };
            displayResults(result.predictions);
            document.getElementById('download-report').classList.remove('hidden');
        } else {
            throw new Error('Prediction failed on the server.');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showConnectionError(resultsContainer, error.message);
    } finally {
        resetButton(btnText, spinner, submitBtn);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// DISPLAY RESULTS — probability bars + explanation tags
// ─────────────────────────────────────────────────────────────────────────────
function displayResults(predictions) {
    const container = document.getElementById('results-container');
    container.innerHTML = '';

    let delay = 0;

    for (const [disease, info] of Object.entries(predictions)) {
        const label       = disease.replace(/_/g, ' ');
        const icon        = DISEASE_ICONS[disease] || '🔬';
        const probPercent = (info.probability * 100).toFixed(1);
        const riskClass   = info.risk_level.toLowerCase();

        // Explanation tags
        const explanationHtml = (info.explanation && info.explanation.length > 0)
            ? `<div class="exp-label">Key Factors:</div>
               <div class="explanation-container">
               ${info.explanation.map(exp => `
                <div class="explanation-tag ${exp.impact}">
                    ${formatFeatureName(exp.feature)}
                    <span class="val">${exp.impact === 'increases' ? '▲' : '▼'}${Math.abs(exp.value).toFixed(1)}%</span>
                </div>`).join('')}
               </div>`
            : `<div class="exp-label" style="color:var(--text-muted)">
                   Result reflects combined clinical profile.
               </div>`;

        const card = document.createElement('div');
        card.className = `disease-card risk-${riskClass}`;
        card.style.animationDelay = `${delay}s`;

        card.innerHTML = `
            <div class="disease-header">
                <span class="disease-name">${icon} ${label}</span>
                <span class="risk-badge risk-${riskClass}">${info.risk_level} Risk</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar prob-bar-animated" style="width:0%" data-target="${probPercent}%"></div>
            </div>
            <div class="prob-text">${probPercent}% probability</div>
            ${explanationHtml}`;

        container.appendChild(card);

        // Animate the bar — double rAF ensures the 0% start is painted first
        const bar = card.querySelector('.prob-bar');
        const targetWidth = probPercent + '%';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 50 + delay * 1000);
            });
        });

        delay += 0.18;
    }
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function resetButton(btnText, spinner, btn) {
    btnText.textContent = 'Analyze Risk';
    btn.querySelector('svg').style.display = '';
    spinner.classList.add('hidden');
    btn.disabled = false;
}

function showValidationError(container) {
    container.innerHTML = `
        <div class="empty-state" style="color:var(--risk-medium)">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <p><strong>Missing Fields</strong><br>Please fill in <em>all</em> patient data fields before analyzing.<br>
            <small style="opacity:0.6">Every field is required for an accurate prediction.</small></p>
        </div>`;
}

function showConnectionError(container, message) {
    container.innerHTML = `
        <div class="empty-state" style="color:var(--risk-high)">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <p><strong>Connection Error</strong><br>${message}<br>
            <small style="opacity:0.6">Make sure the API server is running at ${API_BASE}<br>
            Double-click <strong>run_project.bat</strong> to start it.</small></p>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF REPORT
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('download-report').addEventListener('click', () => {
    if (!lastPrediction) return;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: 'mm', format: 'a4' });
    const p   = lastPrediction.data;
    const res = lastPrediction.predictions;
    const now = new Date().toLocaleString();
    const W = 210, M = 18;
    let y = 0;

    // Header bar
    doc.setFillColor(10, 12, 20);
    doc.rect(0, 0, W, 44, 'F');
    doc.setFillColor(99, 102, 241);
    doc.rect(0, 44, W, 2, 'F');
    doc.setTextColor(240, 242, 248);
    doc.setFontSize(20); doc.setFont('helvetica', 'bold');
    doc.text('MedPredict AI', M, 18);
    doc.setFontSize(10); doc.setFont('helvetica', 'normal');
    doc.setTextColor(139, 149, 176);
    doc.text('Multi-Disease Risk Assessment Report', M, 27);
    doc.text(`Generated: ${now}`, M, 35);

    y = 58;
    doc.setTextColor(30, 30, 30);
    doc.setFontSize(13); doc.setFont('helvetica', 'bold');
    doc.text('Patient Information', M, y);
    y += 2;
    doc.setDrawColor(99, 102, 241); doc.setLineWidth(0.5);
    doc.line(M, y, W - M, y); y += 7;

    const fields = [
        ['Age', p.Age + ' years'], ['Gender', p.Gender],
        ['BMI', p.BMI], ['Systolic BP', `${p.Blood_Pressure_Systolic} mmHg`],
        ['Diastolic BP', `${p.Blood_Pressure_Diastolic} mmHg`],
        ['Fasting Glucose', `${p.Glucose} mg/dL`], ['HbA1c', `${p.HbA1c}%`],
        ['Total Cholesterol', `${p.Cholesterol} mg/dL`],
        ['HDL', `${p.HDL_Cholesterol} mg/dL`], ['LDL', `${p.LDL_Cholesterol} mg/dL`],
        ['Triglycerides', `${p.Triglycerides} mg/dL`], ['Creatinine', `${p.Creatinine} mg/dL`],
        ['ALT', `${p.ALT} U/L`], ['AST', `${p.AST} U/L`],
        ['Smoking', p.Smoking ? 'Yes' : 'No'],
        ['Physical Activity', `Level ${p.Physical_Activity}`],
        ['Alcohol Use', ['None','Moderate','Heavy'][p.Alcohol_Use]],
        ['Family Hx Diabetes', p.Family_History_Diabetes ? 'Yes' : 'No'],
        ['Family Hx Heart', p.Family_History_Heart ? 'Yes' : 'No'],
    ];

    const colW = (W - M * 2) / 2;
    doc.setFontSize(9); doc.setFont('helvetica', 'normal');
    fields.forEach(([label, val], i) => {
        const col = i % 2, row = Math.floor(i / 2);
        const xPos = M + col * colW, yPos = y + row * 6;
        doc.setFont('helvetica', 'bold'); doc.setTextColor(80, 80, 80);
        doc.text(label + ':', xPos, yPos);
        doc.setFont('helvetica', 'normal'); doc.setTextColor(30, 30, 30);
        doc.text(String(val), xPos + 38, yPos);
    });

    y += Math.ceil(fields.length / 2) * 6 + 12;
    doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(30, 30, 30);
    doc.text('Diagnostic Assessment', M, y);
    y += 2; doc.setDrawColor(99, 102, 241);
    doc.line(M, y, W - M, y); y += 8;

    const riskColors = { Low:[52,211,153], Medium:[251,191,36], High:[248,113,113] };

    for (const [disease, info] of Object.entries(res)) {
        const label = disease.replace(/_/g, ' ');
        const prob  = (info.probability * 100).toFixed(1);
        const [r,g,b] = riskColors[info.risk_level] || [150,150,150];

        doc.setFont('helvetica', 'bold'); doc.setFontSize(11); doc.setTextColor(30, 30, 30);
        doc.text(label, M, y);
        doc.setFillColor(r,g,b);
        doc.roundedRect(W - M - 38, y - 5, 38, 7, 2, 2, 'F');
        doc.setTextColor(255,255,255); doc.setFontSize(8);
        doc.text(`${info.risk_level.toUpperCase()}  ${prob}%`, W - M - 36, y - 0.2);
        y += 7;

        doc.setFillColor(220,220,230);
        doc.roundedRect(M, y - 3.5, W - M * 2, 4, 1, 1, 'F');
        const barW = Math.max(2, (W - M * 2) * info.probability);
        doc.setFillColor(r,g,b);
        doc.roundedRect(M, y - 3.5, barW, 4, 1, 1, 'F');
        y += 8;

        if (info.explanation && info.explanation.length > 0) {
            doc.setFont('helvetica', 'normal'); doc.setFontSize(8.5); doc.setTextColor(100,100,100);
            const expText = info.explanation
                .map(e => `${formatFeatureName(e.feature)} (${e.impact === 'increases' ? '+' : '-'}${Math.abs(e.value).toFixed(1)}%)`)
                .join('  |  ');
            doc.text('Key Factors: ' + expText, M, y, { maxWidth: W - M * 2 });
        }
        y += 10;
    }

    y = Math.max(y + 5, 265);
    doc.setFontSize(7.5); doc.setTextColor(160,160,160); doc.setFont('helvetica', 'italic');
    doc.text(
        'DISCLAIMER: This report is for research/educational purposes only. Not a medical diagnosis. Consult a qualified healthcare professional.',
        M, y, { maxWidth: W - M * 2 }
    );
    doc.setDrawColor(200,200,210); doc.setLineWidth(0.3);
    doc.line(M, 282, W - M, 282);
    doc.setFontSize(7.5); doc.setFont('helvetica', 'normal'); doc.setTextColor(160,160,160);
    doc.text('MedPredict AI v3.0', M, 287);
    doc.text(now, W - M - 50, 287);

    doc.save(`MedPredict_Report_${p.Gender}_Age${p.Age}.pdf`);
});

// ─────────────────────────────────────────────────────────────────────────────
// API health check on page load
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            console.log('API server is reachable at', API_BASE);
        } else {
            console.warn('API server returned non-OK status:', res.status);
        }
    } catch {
        console.warn('API server not reachable at', API_BASE, '— run_project.bat to start it.');
    }
});
