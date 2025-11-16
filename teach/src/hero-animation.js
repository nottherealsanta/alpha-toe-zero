/**
 * Hero Animation
 * Displays a background animation of X and O tiles.
 */
export function initHeroAnimation() {
    const container = document.getElementById('hero-bg');
    if (!container) {
        console.error('hero-bg container not found');
        return;
    }

    // Clear existing content
    container.innerHTML = '';

    // Setup hero background tiles
    const fragment = document.createDocumentFragment();
    // Calculate how many tiles we need to fill the viewport (using 40px tiles)
    const tilesX = Math.ceil(window.innerWidth / 40);
    const tilesY = Math.ceil(window.innerHeight / 40);
    const totalTiles = tilesX * tilesY;

    for (let i = 0; i < totalTiles; i++) {
        const tile = document.createElement('div');
        tile.className = 'tile';
        fragment.appendChild(tile);
    }
    container.appendChild(fragment);
    startHeroAnimation();
}

/**
 * Start hero background animation
 */
function startHeroAnimation() {
    const tiles = document.querySelectorAll('.tile');
    const targetFillPercentage = 0.35; // 35% of tiles should be filled
    const targetCount = Math.floor(tiles.length * targetFillPercentage);

    const animate = () => {
        const filledTiles = Array.from(tiles).filter(tile => tile.classList.contains('x') || tile.classList.contains('o'));
        const emptyTiles = Array.from(tiles).filter(tile => !tile.classList.contains('x') && !tile.classList.contains('o'));

        const currentCount = filledTiles.length;

        if (currentCount < targetCount && emptyTiles.length > 0) {
            // Add a new X or O
            const randomTile = emptyTiles[Math.floor(Math.random() * emptyTiles.length)];
            const isX = Math.random() < 0.5;
            randomTile.classList.add(isX ? 'x' : 'o');
        } else if (currentCount > 0) {
            // Randomly remove an X or O
            const randomFilledTile = filledTiles[Math.floor(Math.random() * filledTiles.length)];
            randomFilledTile.className = 'tile';
        }

        setTimeout(animate, 400); // Animate every 400ms
    };

    animate();
}