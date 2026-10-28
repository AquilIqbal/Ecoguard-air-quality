/**
 * Location.js - GPS Location Detection
 */

function getMyLocation() {
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by your browser');
        return;
    }

    const btn = document.querySelector('.btn-location');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Locating...';
    btn.disabled = true;

    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            
            getCityFromCoordinates(lat, lon)
                .then(city => {
                    if (city) {
                        window.location.href = `/?city=${city.toLowerCase()}`;
                    } else {
                        alert('Could not determine city from your location');
                        btn.innerHTML = originalText;
                        btn.disabled = false;
                    }
                })
                .catch(error => {
                    console.error('Error getting city:', error);
                    alert('Error determining your location');
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                });
        },
        function(error) {
            console.error('Geolocation error:', error);
            
            let message = 'Unable to get your location. ';
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    message += 'Please allow location access.';
                    break;
                case error.POSITION_UNAVAILABLE:
                    message += 'Location information unavailable.';
                    break;
                case error.TIMEOUT:
                    message += 'Location request timed out.';
                    break;
                default:
                    message += 'Unknown error occurred.';
            }
            
            alert(message);
            btn.innerHTML = originalText;
            btn.disabled = false;
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );
}

function getCityFromCoordinates(lat, lon) {
    return fetch(`/get-city?lat=${lat}&lon=${lon}`)
        .then(response => response.json())
        .then(data => data.city)
        .catch(error => {
            console.error('Error in reverse geocoding:', error);
            return null;
        });
}

function updateLocationDisplay(city) {
    const display = document.getElementById('current-location');
    if (display) {
        display.innerHTML = `<i class="fas fa-map-marker-alt"></i> ${city}, India`;
    }
}

window.getMyLocation = getMyLocation;
window.updateLocationDisplay = updateLocationDisplay;