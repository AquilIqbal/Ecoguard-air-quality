function toggleLegend() {
    const legend = document.getElementById("aqiLegend");
    const arrow = document.getElementById("legendArrow");

    legend.classList.toggle("collapsed");
    arrow.textContent = legend.classList.contains("collapsed") ? "▸" : "▾";
}
