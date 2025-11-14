// components-animation.js
// Handles inlining and continuous animation for Neural Network and Alpha-MCTS SVGs
// Auto-executed on import.

// Utility: animate stroke draw manually (re-usable)
export function animateStroke(path, duration = 900, emphasis = true) {
  const len = path.getTotalLength();
  path.style.strokeDasharray = len;
  path.style.strokeDashoffset = len;
  if (emphasis) {
    path.dataset._origWidth = path.dataset._origWidth || path.style.strokeWidth || '1.4';
    path.style.transition = 'stroke-width 180ms linear, stroke-opacity 300ms';
    path.style.strokeWidth = (parseFloat(path.dataset._origWidth) + 0.8) + 'px';
    path.style.strokeOpacity = '1';
  }
  return new Promise(resolve => {
    let start;
    function frame(ts) {
      if (!start) start = ts;
      const prog = Math.min((ts - start) / duration, 1);
      path.style.strokeDashoffset = (1 - prog) * len;
      if (prog < 1) {
        requestAnimationFrame(frame);
      } else {
        setTimeout(() => {
          if (emphasis) {
            path.style.strokeWidth = path.dataset._origWidth;
            path.style.strokeOpacity = '0.45';
          }
          resolve();
        }, 250);
      }
    }
    requestAnimationFrame(frame);
  });
}

// ---- Neural Network ----
(function inlineNeuralNet() {
  const selector = '.info-card .card-image img[alt="Neural Network"]';
  const img = document.querySelector(selector);
  if (!img) return;
  fetch(img.src).then(r => r.text()).then(svgText => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = svgText.trim();
    const svg = wrapper.querySelector('svg');
    if (!svg) return;
    svg.removeAttribute('width'); svg.removeAttribute('height');

    // Initial prune (do not animate yet)
    const allPaths = Array.from(svg.querySelectorAll('path'));
    const keepProbability = 0.5;
    const kept = [];
    allPaths.forEach(p => {
      if (Math.random() > keepProbability) { p.remove(); return; }
      kept.push(p);
    });

    // Pause CSS animations until in view
    svg.querySelectorAll('path').forEach(p => { p.style.animationPlayState = 'paused'; });
    svg.querySelectorAll('circle').forEach(c => { c.style.animationPlayState = 'paused'; });

    img.parentNode.replaceChild(svg, img);

    // Start animations only when SVG enters viewport
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Resume CSS animations and apply randomization
          const leftBase = 900;
          const midBase = 2500;
          kept.forEach(p => {
            p.classList.add('conn-anim');
            const isMidRight = p.classList.contains('m');
            const base = isMidRight ? midBase : leftBase;
            const rnd = Math.random() * 1200;
            p.style.animationDelay = (base + rnd) / 1000 + 's';
            const w = 0.9 + Math.random() * 0.9;
            p.style.strokeWidth = w.toFixed(2) + 'px';
            p.style.animationPlayState = 'running';
          });
          svg.querySelectorAll('circle').forEach(c => { c.style.animationPlayState = 'running'; });
          setTimeout(() => startNeuralDynamics(svg), 5000);
          observer.unobserve(svg);
        }
      });
    }, { threshold: 0.1 });
    observer.observe(svg);
  }).catch(() => { });
})();

function startNeuralDynamics(svg) {
  const MAX_DYNAMIC = 60;
  const dynamicEdges = [];
  const existingKey = new Set();

  // Collect nodes by layer via x coordinate buckets
  const circles = Array.from(svg.querySelectorAll('circle.node'));
  const left = circles.filter(c => parseFloat(c.getAttribute('cx')) < 60);
  const mid = circles.filter(c => parseFloat(c.getAttribute('cx')) > 60 && parseFloat(c.getAttribute('cx')) < 180);
  const right = circles.filter(c => parseFloat(c.getAttribute('cx')) > 180);

  [left, mid, right].forEach((arr, i) => arr.forEach((c, idx) => { c.dataset.nnId = ['L', 'M', 'R'][i] + idx; }));

  function edgeKey(a, b) { return a.dataset.nnId + '>' + b.dataset.nnId; }

  function makeCurvedPath(a, b) {
    const ax = parseFloat(a.getAttribute('cx')); const ay = parseFloat(a.getAttribute('cy'));
    const bx = parseFloat(b.getAttribute('cx')); const by = parseFloat(b.getAttribute('cy'));
    const mx = (ax + bx) / 2; const my = (ay + by) / 2 + (Math.random() * 30 - 15);
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', `M ${ax} ${ay} Q ${mx} ${my} ${bx} ${by}`);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', 'var(--conn-svg)');
    p.style.strokeOpacity = '.55';
    p.style.strokeWidth = (0.8 + Math.random() * 1.4).toFixed(2) + 'px';
    p.classList.add('dynamic-edge');
    return p;
  }

  function addEdge() {
    const mode = Math.random() < 0.5 ? 'LM' : 'MR';
    const fromArr = mode === 'LM' ? left : mid;
    const toArr = mode === 'LM' ? mid : right;
    if (!fromArr.length || !toArr.length) return;
    const a = fromArr[Math.floor(Math.random() * fromArr.length)];
    const b = toArr[Math.floor(Math.random() * toArr.length)];
    const k = edgeKey(a, b);
    if (existingKey.has(k)) return;
    existingKey.add(k);
    const path = makeCurvedPath(a, b);
    svg.firstElementChild.appendChild(path);
    dynamicEdges.push(path);
    animateStroke(path, 1000, false);
    if (dynamicEdges.length > MAX_DYNAMIC) {
      const old = dynamicEdges.shift();
      old.style.transition = 'opacity 600ms';
      old.style.opacity = '0';
      setTimeout(() => { existingKey.delete(edgeKey(old._a, old._b)); old.remove(); }, 700);
    }
  }

  function modulateThickness() {
    const all = svg.querySelectorAll('path');
    all.forEach(p => {
      if (!p.dataset.thicknessTransition) {
        p.style.transition = 'stroke-width 1600ms ease-in-out, stroke-opacity 1600ms ease-in-out';
        p.dataset.thicknessTransition = '1';
      }
      const base = p.classList.contains('dynamic-edge') ? 1.0 : 0.9;
      const target = (base + Math.random() * 1.6).toFixed(2) + 'px';
      p.style.strokeWidth = target;
      const opBase = p.classList.contains('dynamic-edge') ? 0.55 : 0.45;
      p.style.strokeOpacity = (opBase + Math.random() * 0.35).toFixed(2);
    });
  }

  svg.querySelectorAll('path').forEach(p => { p.style.transition = 'stroke-width 1600ms ease-in-out, stroke-opacity 1600ms ease-in-out'; });
  setInterval(addEdge, 900);
  setInterval(modulateThickness, 2200);
}

// ---- Alpha MCTS ----
(function inlineAlphaMCTS() {
  const selector = '.info-card .card-image img[alt="αMCTS — Alpha Monte Carlo Tree Search"]';
  const img = document.querySelector(selector);
  if (!img) return;
  fetch(img.src).then(r => r.text()).then(svgText => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = svgText.trim();
    const svg = wrapper.querySelector('svg');
    if (!svg) return;
    svg.removeAttribute('width'); svg.removeAttribute('height');
    
    // Pause CSS animations until in view
    svg.querySelectorAll('path').forEach(p => { p.style.animationPlayState = 'paused'; });
    svg.querySelectorAll('circle').forEach(c => { c.style.animationPlayState = 'paused'; });
    
    img.parentNode.replaceChild(svg, img);
    
    // Start animations only when SVG enters viewport
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          svg.querySelectorAll('path').forEach(p => { p.style.animationPlayState = 'running'; });
          svg.querySelectorAll('circle').forEach(c => { c.style.animationPlayState = 'running'; });
          setTimeout(() => startMCTSTraversal(svg), 400);
          observer.unobserve(svg);
        }
      });
    }, { threshold: 0.1 });
    observer.observe(svg);
  }).catch(() => { });
})();

function startMCTSTraversal(svg) {
  const paths = Array.from(svg.querySelectorAll('path.draw'));
  const rootLeft = paths.find(p => /M94 8.* 44 76/.test(p.getAttribute('d')));
  const rootRight = paths.find(p => /M94 8.* 148 76/.test(p.getAttribute('d')));
  const leftChildren = paths.filter(p => /^M44 76 /.test(p.getAttribute('d')));
  const rightChildren = paths.filter(p => /^M148 76 /.test(p.getAttribute('d')));
  const deeper = paths.filter(p => /^M74 138 /.test(p.getAttribute('d')));

  async function cycle() {
    if (rootLeft) { await animateStroke(rootLeft, 1000); }
    for (const c of leftChildren) { await animateStroke(c, 800); }
    if (rootLeft) { await animateStroke(rootLeft, 600, false); }
    if (rootRight) { await animateStroke(rootRight, 1000); }
    for (const c of rightChildren) { await animateStroke(c, 800); }
    if (deeper.length) { for (const d of deeper) { await animateStroke(d, 700); } }
    if (rootRight) { await animateStroke(rootRight, 600, false); }
    setTimeout(cycle, 900);
  }
  cycle();
}

// ---- Self-Play ----
(function inlineSelfPlay() {
  const selector = '.info-card .card-image img[alt="Self-Play"]';
  const img = document.querySelector(selector);
  if (!img) return;
  fetch(img.src).then(r => r.text()).then(svgText => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = svgText.trim();
    const svg = wrapper.querySelector('svg');
    if (!svg) return;
    svg.removeAttribute('width'); svg.removeAttribute('height');
    
    // Pause all animations initially (include moving agent)
    svg.querySelectorAll('.move, .arrow, .loop-arrow, .agent').forEach(el => {
      el.style.animationPlayState = 'paused';
    });
    
    img.parentNode.replaceChild(svg, img);
    
    // Start animations only when SVG enters viewport
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Reset all animations to start from beginning
          svg.querySelectorAll('.move, .arrow, .loop-arrow, .agent').forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight; // Trigger reflow
            el.style.animation = null;
          });
          
          // Resume all animations (include moving agent)
          svg.querySelectorAll('.move, .arrow, .loop-arrow, .agent').forEach(el => {
            el.style.animationPlayState = 'running';
          });
          
          // Stop animations after 2 complete cycles (30 seconds = 2 × 15s cycle)
          setTimeout(() => {
            svg.querySelectorAll('.move, .arrow, .loop-arrow, .agent').forEach(el => {
              el.style.animationPlayState = 'paused';
            });
          }, 30000);
          
          observer.unobserve(svg);
        }
      });
    }, { threshold: 0.1 });
    observer.observe(svg);
  }).catch(() => { });
})();
