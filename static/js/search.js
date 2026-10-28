/**
 * Search.js - City Search Autocomplete
 */

document.addEventListener('DOMContentLoaded', function() {
    setupSearch();
});

function setupSearch() {
    const searchInput = document.getElementById('city-search');
    if (!searchInput) return;

    let timeout = null;
    let resultsDiv = null;
    let visibleCities = [];
    let activeIndex = -1;

    function createResultsDropdown() {
        if (resultsDiv) return;

        resultsDiv = document.createElement('div');
        resultsDiv.className = 'search-results';
        resultsDiv.style.height = 'auto';
        resultsDiv.style.minHeight = '0';
        resultsDiv.style.backdropFilter = 'none';
        resultsDiv.style.webkitBackdropFilter = 'none';
        searchInput.parentElement.classList.add('search-form');
        searchInput.parentElement.appendChild(resultsDiv);
    }

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeRegExp(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function highlightMatch(city, query) {
        const safeCity = escapeHtml(city);
        if (!query) return safeCity;

        const regex = new RegExp(`(${escapeRegExp(query)})`, 'ig');
        return safeCity.replace(regex, '<span class="search-match">$1</span>');
    }

    function hideResults() {
        resultsDiv.style.display = 'none';
        resultsDiv.innerHTML = '';
        visibleCities = [];
        activeIndex = -1;
    }

    function selectCity(city) {
        if (!city) return;
        window.location.href = `/?city=${city.toLowerCase()}`;
    }

    function setActiveItem(index) {
        const items = resultsDiv.querySelectorAll('.search-result-item');
        items.forEach(item => item.classList.remove('active'));

        if (index < 0 || index >= items.length) {
            activeIndex = -1;
            return;
        }

        activeIndex = index;
        items[activeIndex].classList.add('active');
        items[activeIndex].scrollIntoView({ block: 'nearest' });
    }

    createResultsDropdown();

    searchInput.addEventListener('input', function() {
        clearTimeout(timeout);
        const query = this.value.trim();

        if (query.length < 2) {
            hideResults();
            return;
        }

        timeout = setTimeout(() => {
            fetch(`/search-city?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(cities => {
                    if (cities.length > 0) {
                        showResults(cities, query);
                    } else {
                        hideResults();
                    }
                })
                .catch(err => {
                    console.error('Search error:', err);
                    hideResults();
                });
        }, 300);
    });

    function showResults(cities, query) {
        resultsDiv.innerHTML = '';
        const validCities = cities.filter(city => typeof city === 'string' && city.trim() !== '');

        if (validCities.length === 0) {
            hideResults();
            return;
        }

        visibleCities = validCities;
        activeIndex = -1;

        validCities.forEach(city => {
            const item = document.createElement('div');
            item.className = 'search-result-item';
            item.innerHTML = highlightMatch(city, query);

            item.addEventListener('click', function() {
                selectCity(city);
            });

            item.addEventListener('mouseenter', function() {
                const idx = visibleCities.indexOf(city);
                if (idx !== -1) setActiveItem(idx);
            });
            
            resultsDiv.appendChild(item);
        });
        
        resultsDiv.style.display = 'block';
    }

    searchInput.addEventListener('keydown', function(e) {
        const isOpen = resultsDiv.style.display === 'block' && visibleCities.length > 0;

        if (e.key === 'Escape') {
            hideResults();
            return;
        }

        if (!isOpen) {
            if (e.key === 'Enter') {
                const form = this.closest('form');
                if (form) form.submit();
            }
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            const next = activeIndex < visibleCities.length - 1 ? activeIndex + 1 : 0;
            setActiveItem(next);
            return;
        }

        if (e.key === 'ArrowUp') {
            e.preventDefault();
            const prev = activeIndex > 0 ? activeIndex - 1 : visibleCities.length - 1;
            setActiveItem(prev);
            return;
        }

        if (e.key === 'Enter') {
            e.preventDefault();
            if (activeIndex >= 0) {
                selectCity(visibleCities[activeIndex]);
            } else {
                selectCity(visibleCities[0]);
            }
        }
    });

    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !resultsDiv.contains(e.target)) {
            hideResults();
        }
    });
}
