/**
 * Chart.js - AQI Trend & Forecast Chart
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    setupChartTabs();
});

let aqiChart = null;
let chartBase = {
    labels: [],
    official: [],
    ai: []
};
let activeTab = 'official';
let officialForecastAvailable = true;

function initializeChart() {
    const canvas = document.getElementById('aqi-chart');
    if (!canvas) {
        console.log('Chart canvas not found');
        return;
    }

    if (aqiChart) {
        aqiChart.destroy();
    }

    const data = window.chartData || {};
    const labels = Array.isArray(data.labels) ? data.labels : [];
    const official = Array.isArray(data.actual) ? data.actual : [];
    const ai = Array.isArray(data.predicted) ? data.predicted : [];
    officialForecastAvailable = data.officialForecastAvailable !== false;

    chartBase = normalizeBaseData(labels, official, ai);
    alignAiSeriesToCards(chartBase, data);

    aqiChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: buildCommonOptions()
    });

    applyChartMode('official');
}

function alignAiSeriesToCards(base, raw) {
    if (!base || !Array.isArray(base.labels) || !Array.isArray(base.ai)) return;

    const currentAqi = Number(raw.currentAqi);
    const tomorrowAqi = Number(raw.tomorrowAqi);
    const todayLabel = raw.todayLabel;
    const tomorrowLabel = raw.tomorrowLabel;

    const todayIdx = base.labels.indexOf(todayLabel);
    const tomorrowIdx = base.labels.indexOf(tomorrowLabel);

    if (Number.isFinite(currentAqi) && todayIdx >= 0) {
        base.ai[todayIdx] = clampAqi(currentAqi);
    }
    if (Number.isFinite(tomorrowAqi) && tomorrowIdx >= 0) {
        base.ai[tomorrowIdx] = clampAqi(tomorrowAqi);
    }
}

function normalizeBaseData(labels, official, ai) {
    const safeLabels = labels.length ? labels : ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
    const len = safeLabels.length;

    const safeOfficial = Array.from({ length: len }, (_, i) => clampAqi(Number(official[i] || 0)));
    const safeAi = Array.from({ length: len }, (_, i) => clampAqi(Number(ai[i] || safeOfficial[i] || 0)));

    return {
        labels: safeLabels,
        official: safeOfficial,
        ai: safeAi
    };
}

function clampAqi(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(500, Math.round(value)));
}

function buildCommonOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(10, 18, 48, 0.95)',
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                borderColor: 'rgba(255, 255, 255, 0.18)',
                borderWidth: 1,
                padding: 12,
                displayColors: true,
                callbacks: {
                    label: function(context) {
                        const value = clampAqi(context.parsed.y);
                        const category = getAqiCategory(value);
                        return `${context.dataset.label}: ${value} AQI (${category})`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                suggestedMax: 300,
                ticks: {
                    color: 'rgba(255, 255, 255, 0.9)',
                    font: { size: 12, weight: 500 },
                    callback: function(value) {
                        return `${value} AQI`;
                    }
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.09)',
                    drawBorder: false
                }
            },
            x: {
                ticks: {
                    color: 'rgba(255, 255, 255, 0.9)',
                    font: { size: 12, weight: 500 }
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.04)',
                    drawBorder: false
                }
            }
        }
    };
}

function getBarColor(value, alpha = 0.9) {
    if (value <= 50) return `rgba(16, 185, 129, ${alpha})`;
    if (value <= 100) return `rgba(245, 158, 11, ${alpha})`;
    if (value <= 200) return `rgba(249, 115, 22, ${alpha})`;
    if (value <= 300) return `rgba(239, 68, 68, ${alpha})`;
    return `rgba(168, 85, 247, ${alpha})`;
}

function getAqiCategory(value) {
    if (value <= 50) return 'Good';
    if (value <= 100) return 'Satisfactory';
    if (value <= 200) return 'Moderate';
    if (value <= 300) return 'Poor';
    if (value <= 400) return 'Very Poor';
    return 'Severe';
}

function getMaxForSeries(seriesList) {
    const maxValue = Math.max(...seriesList.flat(), 50);
    const padded = Math.ceil((maxValue + 20) / 50) * 50;
    return Math.min(500, Math.max(200, padded));
}

function setLegend(items) {
    const legendEl = document.getElementById('chart-legend');
    if (!legendEl) return;

    legendEl.innerHTML = items
        .map(item => (
            `<span class="legend-item">
                <span class="legend-dot" style="background: ${item.color};"></span>
                ${item.label}
             </span>`
        ))
        .join('');
}

function setChartEmptyState(show, message) {
    const emptyStateEl = document.getElementById('chart-empty-state');
    if (!emptyStateEl) return;
    emptyStateEl.textContent = message || '';
    emptyStateEl.classList.toggle('visible', !!show);
}

function applyChartMode(mode) {
    if (!aqiChart) return;
    activeTab = mode;

    if (mode === 'official') {
        const values = chartBase.official;
        setChartEmptyState(
            !officialForecastAvailable,
            'Official WAQI forecast is unavailable for this city right now. Use AI Forecast or Compare for the current projection.'
        );
        aqiChart.config.type = 'bar';
        aqiChart.data.labels = chartBase.labels;
        aqiChart.data.datasets = [
            {
                label: 'Official Forecast',
                data: values,
                backgroundColor: values.map(v => getBarColor(v, 0.9)),
                borderColor: values.map(v => getBarColor(v, 1)),
                borderWidth: 1.5,
                borderRadius: 8,
                borderSkipped: false,
                barPercentage: 0.62,
                categoryPercentage: 0.72
            }
        ];
        aqiChart.options.scales.y.suggestedMax = getMaxForSeries([values]);
        setLegend([
            {
                label: officialForecastAvailable ? 'Official Forecast (WAQI)' : 'Current AQI only (forecast unavailable)',
                color: '#f97316'
            }
        ]);
    } else if (mode === 'ai') {
        const values = chartBase.ai;
        setChartEmptyState(false);
        aqiChart.config.type = 'bar';
        aqiChart.data.labels = chartBase.labels;
        aqiChart.data.datasets = [
            {
                label: 'EcoGuard AI Forecast',
                data: values,
                backgroundColor: values.map(v => getBarColor(v, 0.82)),
                borderColor: 'rgba(59, 130, 246, 0.95)',
                borderWidth: 1.5,
                borderRadius: 8,
                borderSkipped: false,
                barPercentage: 0.62,
                categoryPercentage: 0.72
            }
        ];
        aqiChart.options.scales.y.suggestedMax = getMaxForSeries([values]);
        setLegend([{ label: 'EcoGuard AI Forecast', color: '#3b82f6' }]);
    } else {
        const official = chartBase.official;
        const ai = chartBase.ai;
        setChartEmptyState(false);
        aqiChart.config.type = 'line';
        aqiChart.data.labels = chartBase.labels;
        aqiChart.data.datasets = [
            {
                label: 'Official Forecast',
                data: official,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.08)',
                borderWidth: 3,
                fill: false,
                tension: 0.32,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#ef4444',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            },
            {
                label: 'EcoGuard AI Forecast',
                data: ai,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                borderWidth: 3,
                borderDash: [10, 5],
                fill: false,
                tension: 0.32,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }
        ];
        aqiChart.options.scales.y.suggestedMax = getMaxForSeries([official, ai]);
        setLegend([
            { label: 'Official Forecast (WAQI)', color: '#ef4444' },
            { label: 'EcoGuard AI Forecast', color: '#3b82f6' }
        ]);
    }

    aqiChart.update();
}

function setupChartTabs() {
    const chartCanvas = document.getElementById('aqi-chart');
    if (!chartCanvas) return;

    const chartCard = chartCanvas.closest('.card');
    if (!chartCard) return;

    const tabButtons = chartCard.querySelectorAll('.card-header .tab-btn');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.dataset.tab;
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            applyChartMode(tab);
        });
    });
}

window.ChartModule = {
    initialize: initializeChart,
    setMode: applyChartMode,
    getActiveMode: function() { return activeTab; }
};
