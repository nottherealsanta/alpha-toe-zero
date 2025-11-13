import * as THREE from 'three';

/**
 * 3D Qubic Animation
 * Displays an interactive 4x4x4 cube with animated winning lines
 */

export function initQubicAnimation() {
    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    function init() {
        const container = document.getElementById('qubic-animation');
        if (!container) {
            console.error('qubic-animation container not found');
            return;
        }
        
        console.log('Initializing Qubic animation...', container.clientWidth, 'x', container.clientHeight);

        let scene, camera, renderer;
        let lines = [];
        let isAnimating = true;
        let lineOrder = [];
        
        const GRID_SIZE = 4;
        const SPACING = 2;
        
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        let rotation = { x: 0, y: 0 };
        
        function initScene() {
            scene = new THREE.Scene();
            
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(8, 8, 8);
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.domElement.id = 'qubic-animation-canvas';
            container.appendChild(renderer.domElement);
            
            console.log('Renderer created and added to container');

            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'qubic-controls';
            container.appendChild(controlsDiv);
            
            const ambientLight = new THREE.AmbientLight(0xffffff, 1);
            scene.add(ambientLight);
            
            createGridStructure();
            createWinningLines();
            
            lineOrder = Array.from({length: lines.length}, (_, i) => i);
            shuffleArray(lineOrder);
            
            container.addEventListener('mousedown', onMouseDown);
            container.addEventListener('mousemove', onMouseMove);
            container.addEventListener('mouseup', onMouseUp);
            container.addEventListener('mouseleave', onMouseUp);
            window.addEventListener('resize', onWindowResize);
            
            animate();
        }

        function animate() {
            requestAnimationFrame(animate);
            
            if (!isDragging) {
                rotation.y += 0.001; // Slow continuous rotation
                rotation.x += 0.0005; // Slow continuous rotation
            }

            scene.rotation.x = rotation.x;
            scene.rotation.y = rotation.y;
            
            if (isAnimating) {
                const cycleTime = 50;
                const currentFrame = Math.floor(Date.now() / 16) % (lines.length * cycleTime);
                const activePosition = Math.floor(currentFrame / cycleTime);
                const activeLineIndex = lineOrder[activePosition];
                const progress = (currentFrame % cycleTime) / cycleTime;
                
                for (let i = 0; i < lines.length; i++) {
                    let opacity = 0;
                    
                    if (i === activeLineIndex) {
                        opacity = Math.sin(progress * Math.PI);
                    }
                    
                    lines[i].material.opacity = opacity;
                }
            }
            
            renderer.render(scene, camera);
        }
        
        function onMouseDown(e) {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
            container.classList.add('grabbing');
        }
        
        function onMouseMove(e) {
            if (!isDragging) return;
            
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            
            rotation.y += deltaX * 0.01;
            rotation.x += deltaY * 0.01;
            
            previousMousePosition = { x: e.clientX, y: e.clientY };
        }
        
        function onMouseUp(e) {
            isDragging = false;
            container.classList.remove('grabbing');
        }
        
        function shuffleArray(array) {
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
        }
        
        function createGridStructure() {
            const offset = -(GRID_SIZE - 1) * SPACING / 2;
            const end = (GRID_SIZE - 1) * SPACING / 2;
            
            const gridGroup = new THREE.Group();
            const gridMaterial = new THREE.LineBasicMaterial({ color: 0xcccccc, linewidth: 3 });
            
            // X-axis grid lines
            for (let y = 0; y < GRID_SIZE; y++) {
                for (let z = 0; z < GRID_SIZE; z++) {
                    const points = [
                        new THREE.Vector3(offset, offset + y * SPACING, offset + z * SPACING),
                        new THREE.Vector3(end, offset + y * SPACING, offset + z * SPACING)
                    ];
                    const geo = new THREE.BufferGeometry().setFromPoints(points);
                    gridGroup.add(new THREE.Line(geo, gridMaterial));
                }
            }
            
            // Y-axis grid lines
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let z = 0; z < GRID_SIZE; z++) {
                    const points = [
                        new THREE.Vector3(offset + x * SPACING, offset, offset + z * SPACING),
                        new THREE.Vector3(offset + x * SPACING, end, offset + z * SPACING)
                    ];
                    const geo = new THREE.BufferGeometry().setFromPoints(points);
                    gridGroup.add(new THREE.Line(geo, gridMaterial));
                }
            }
            
            // Z-axis grid lines
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    const points = [
                        new THREE.Vector3(offset + x * SPACING, offset + y * SPACING, offset),
                        new THREE.Vector3(offset + x * SPACING, offset + y * SPACING, end)
                    ];
                    const geo = new THREE.BufferGeometry().setFromPoints(points);
                    gridGroup.add(new THREE.Line(geo, gridMaterial));
                }
            }
            
            const pointGeometry = new THREE.BufferGeometry();
            const positions = [];
            
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let z = 0; z < GRID_SIZE; z++) {
                        positions.push(
                            offset + x * SPACING,
                            offset + y * SPACING,
                            offset + z * SPACING
                        );
                    }
                }
            }
            
            pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
            const pointMaterial = new THREE.PointsMaterial({
                color: 0xdddddd,
                size: 0.25,
                sizeAttenuation: true
            });
            
            const points = new THREE.Points(pointGeometry, pointMaterial);
            gridGroup.add(points);
            
            scene.add(gridGroup);
        }
        
        function createWinningLines() {
            const offset = -(GRID_SIZE - 1) * SPACING / 2;
            const end = (GRID_SIZE - 1) * SPACING / 2;
            
            // Horizontal lines (along X)
            for (let y = 0; y < GRID_SIZE; y++) {
                for (let z = 0; z < GRID_SIZE; z++) {
                    addLine(
                        offset, offset + y * SPACING, offset + z * SPACING,
                        end, offset + y * SPACING, offset + z * SPACING
                    );
                }
            }
            
            // Vertical lines (along Y)
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let z = 0; z < GRID_SIZE; z++) {
                    addLine(
                        offset + x * SPACING, offset, offset + z * SPACING,
                        offset + x * SPACING, end, offset + z * SPACING
                    );
                }
            }
            
            // Depth lines (along Z)
            for (let x = 0; x < GRID_SIZE; x++) {
                for (let y = 0; y < GRID_SIZE; y++) {
                    addLine(
                        offset + x * SPACING, offset + y * SPACING, offset,
                        offset + x * SPACING, offset + y * SPACING, end
                    );
                }
            }
            
            // Face diagonals (XY plane)
            for (let z = 0; z < GRID_SIZE; z++) {
                addLine(offset, offset, offset + z * SPACING, end, end, offset + z * SPACING);
                addLine(end, offset, offset + z * SPACING, offset, end, offset + z * SPACING);
            }
            
            // Face diagonals (XZ plane)
            for (let y = 0; y < GRID_SIZE; y++) {
                addLine(offset, offset + y * SPACING, offset, end, offset + y * SPACING, end);
                addLine(end, offset + y * SPACING, offset, offset, offset + y * SPACING, end);
            }
            
            // Face diagonals (YZ plane)
            for (let x = 0; x < GRID_SIZE; x++) {
                addLine(offset + x * SPACING, offset, offset, offset + x * SPACING, end, end);
                addLine(offset + x * SPACING, end, offset, offset + x * SPACING, offset, end);
            }
            
            // Space diagonals
            addLine(offset, offset, offset, end, end, end);
            addLine(end, offset, offset, offset, end, end);
            addLine(offset, end, offset, end, offset, end);
            addLine(end, end, offset, offset, offset, end);
        }
        
        function addLine(x1, y1, z1, x2, y2, z2) {
            const points = [
                new THREE.Vector3(x1, y1, z1),
                new THREE.Vector3(x2, y2, z2)
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
                color: 0x0969da,
                linewidth: 5,
                transparent: true,
                opacity: 0
            });
            
            const line = new THREE.Line(geometry, material);
            lines.push(line);
            scene.add(line);
        }
        
        function onWindowResize() {
            if (!container) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
        
        initScene();
    }
}
