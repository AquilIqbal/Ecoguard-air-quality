/**
 * Map.js - Interactive AQI Heatmap
 */

let map = null;
let heatmapLayer = null;
let markersLayer = null;
let currentTileLayer = null;
let currentMapMode = 'city';
const INDIA_BOUNDS = L.latLngBounds(
    [6.0, 67.0],   // SW
    [37.8, 98.8]   // NE
);

document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    setupMapTabs();
});

function initializeMap() {
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.log('Map element not found');
        return;
    }

    const city = (window.mapData && window.mapData.city) ? window.mapData.city : 'delhi';
    
    if (!window.mapData || window.mapData.lat == null || window.mapData.lon == null) {
        console.log('No coordinates available, will load from heatmap data');
        map = L.map('map').setView([20.5937, 78.9629], 5);
        currentTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        });
        currentTileLayer.addTo(map);
        markersLayer = L.layerGroup().addTo(map);
        loadCityHeatmap(city);
        return;
    }

    const lat = window.mapData.lat;
    const lon = window.mapData.lon;

    console.log('Initializing map for', city, 'at', lat, lon);

    map = L.map('map').setView([lat, lon], 11);
    currentTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    });
    currentTileLayer.addTo(map);
    markersLayer = L.layerGroup().addTo(map);
    loadCityHeatmap(city);
}

function loadCityHeatmap(city) {
    console.log('Loading heatmap for:', city);
    
    fetch(`/aqi-heatmap?city=${city}`)
        .then(response => response.json())
        .then(data => {
            console.log('Heatmap data received:', data.length, 'points');
            if (data && data.length > 0) {
                displayHeatmap(data);
                addMarkers(data);
                const hasMapCenter = window.mapData && window.mapData.lat != null && window.mapData.lon != null;
                if (hasMapCenter) {
                    map.setView([window.mapData.lat, window.mapData.lon], 11);
                } else if (data[0] && data[0].lat && data[0].lon) {
                    map.setView([data[0].lat, data[0].lon], 11);
                }
            } else {
                console.log('No heatmap data for', city);
            }
        })
        .catch(error => {
            console.error('Error loading heatmap:', error);
        });
}

function displayHeatmap(data) {
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }

    const heatData = data.map(point => [
        point.lat,
        point.lon,
        point.intensity
    ]);

    console.log('Creating heatmap with', heatData.length, 'points');

    heatmapLayer = L.heatLayer(heatData, {
        radius: 25,
        blur: 35,
        maxZoom: 13,
        gradient: {
            0.0: '#10b981',
            0.3: '#fbbf24',
            0.5: '#f97316',
            0.7: '#ef4444',
            1.0: '#a855f7'
        }
    }).addTo(map);
}

function addMarkers(data) {
    markersLayer.clearLayers();
    console.log('Adding', data.length, 'markers');

    data.forEach(point => {
        let color;
        if (point.aqi <= 50) color = '#10b981';
        else if (point.aqi <= 100) color = '#fbbf24';
        else if (point.aqi <= 200) color = '#f97316';
        else if (point.aqi <= 300) color = '#ef4444';
        else color = '#a855f7';

        const marker = L.circleMarker([point.lat, point.lon], {
            radius: 8,
            fillColor: color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        });

        marker.bindPopup(`
            <div style="text-align: center; font-family: 'Outfit', sans-serif;">
                <strong style="font-size: 1.2em;">${point.aqi}</strong><br>
                <span style="color: ${color}; font-weight: 600;">AQI</span>
            </div>
        `);

        marker.addTo(markersLayer);
    });
}

function loadIndiaHeatmap() {
    console.log('Loading India heatmap');
    
    fetch('/aqi-india-heatmap')
        .then(response => response.json())
        .then(data => {
            console.log('India heatmap data:', data.length, 'points');
            
            if (data && data.length > 0) {
                const cleanData = data
                    .filter(point => point && typeof point === 'object')
                    .map(point => ({
                        lat: Number(point.lat),
                        lon: Number(point.lon),
                        intensity: Number(point.intensity),
                        aqi: Number(point.aqi),
                        name: point.name || 'Unknown location',
                        live: Boolean(point.live)
                    }))
                    .filter(point =>
                        Number.isFinite(point.lat) &&
                        Number.isFinite(point.lon)
                    );

                if (cleanData.length === 0) {
                    console.log('India heatmap data is not in expected format');
                    return;
                }

                // Slightly zoomed-out framing so the full India region is visible on first render.
                map.fitBounds(INDIA_BOUNDS, {
                    padding: [18, 18],
                    maxZoom: 5
                });
                map.setZoom(map.getZoom() + 1);

                if (heatmapLayer) {
                    map.removeLayer(heatmapLayer);
                }

                // Country-level view needs a larger radius/minOpacity to remain visible.
                const heatInput = cleanData
                    .filter(point => point.live && Number.isFinite(point.intensity) && point.intensity > 0)
                    .map(point => [point.lat, point.lon, point.intensity]);

                if (heatInput.length > 0) {
                    heatmapLayer = L.heatLayer(heatInput, {
                        radius: 28,
                        blur: 24,
                        minOpacity: 0.45,
                        maxZoom: 9,
                        max: 1.0,
                        gradient: {
                            0.0: '#10b981',
                            0.3: '#fbbf24',
                            0.5: '#f97316',
                            0.7: '#ef4444',
                            1.0: '#a855f7'
                        }
                    }).addTo(map);
                }

                markersLayer.clearLayers();
                addIndiaMarkers(cleanData);
            }
        })
        .catch(error => {
            console.error('Error loading India heatmap:', error);
        });
}

function addIndiaMarkers(data) {
    if (!markersLayer) return;

    data.forEach(point => {
        const lat = point.lat;
        const lon = point.lon;
        const intensity = point.intensity;
        const aqi = Number.isFinite(point.aqi) ? point.aqi : null;
        const name = point.name || 'Unknown location';
        const isLive = point.live === true;

        let color;
        if (!isLive || aqi === null) color = '#94a3b8';
        else if (aqi <= 50) color = '#10b981';
        else if (aqi <= 100) color = '#fbbf24';
        else if (aqi <= 200) color = '#f97316';
        else if (aqi <= 300) color = '#ef4444';
        else color = '#a855f7';

        L.circleMarker([lat, lon], {
            radius: isLive ? 4 : 3,
            fillColor: color,
            color: '#ffffff',
            weight: 1,
            opacity: isLive ? 0.9 : 0.6,
            fillOpacity: isLive ? 0.8 : 0.5
        })
        .bindTooltip(
            isLive ? `${name}<br>AQI: ${aqi}` : `${name}<br>No live AQI`,
            {
                direction: 'top',
                sticky: true,
                opacity: 0.95
            }
        )
        .on('mouseover', function() {
            this.setStyle({ radius: isLive ? 6 : 5, weight: 2 });
        })
        .on('mouseout', function() {
            this.setStyle({ radius: isLive ? 4 : 3, weight: 1 });
        })
        .addTo(markersLayer);
    });
}

function setupMapTabs() {
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.log('Map element not found for tabs');
        return;
    }

    const mapCard = mapElement.closest('.card');
    if (!mapCard) {
        console.log('Map card not found');
        return;
    }

    const tabButtons = mapCard.querySelectorAll('.card-header .tab-btn');
    console.log('Found', tabButtons.length, 'map tab buttons');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.dataset.tab;
            console.log('Map tab clicked:', tab);
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            handleMapTabChange(tab);
        });
    });
}

function handleMapTabChange(tab) {
    const city = (window.mapData && window.mapData.city) ? window.mapData.city : 'delhi';
    currentMapMode = tab;
    
    switch(tab) {
        case 'city':
            console.log('Switching to city view for', city);
            const lat = (window.mapData && window.mapData.lat != null) ? window.mapData.lat : 28.6139;
            const lon = (window.mapData && window.mapData.lon != null) ? window.mapData.lon : 77.2090;
            map.setView([lat, lon], 11);
            loadCityHeatmap(city);
            break;
            
        case 'india':
            console.log('Switching to India view');
            loadIndiaHeatmap();
            break;
    }
}

window.MapModule = {
    initialize: initializeMap,
    loadCity: loadCityHeatmap,
    loadIndia: loadIndiaHeatmap
};
