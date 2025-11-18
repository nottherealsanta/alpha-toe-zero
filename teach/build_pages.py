import os
import subprocess
from bs4 import BeautifulSoup


def convert_notebook_to_html(notebook_path, output_dir, output_file):
    """Convert a Jupyter notebook to HTML using nbconvert."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output-dir",
            output_dir,
            "--output",
            output_file,
            notebook_path,
        ]
    )


def add_thebe_core_to_html(html_path):
    """Modify the HTML to integrate Thebe Core for interactive code execution."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove all style tags to avoid conflicts
    for style_tag in soup.find_all("style"):
        style_tag.decompose()

    # Add Google Fonts
    soup.head.append(
        soup.new_tag("link", rel="preconnect", href="https://fonts.googleapis.com")
    )
    soup.head.append(
        soup.new_tag(
            "link",
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        )
    )
    gf_href = "https://fonts.googleapis.com/css2?family=Fira+Code:wght@300..700&family=Lora:ital,wght@0,400..700;1,400..700&family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap"
    soup.head.append(soup.new_tag("link", href=gf_href, rel="stylesheet"))

    # Add RequireJS for Jupyter widget support
    require_script = soup.new_tag(
        "script",
        type="text/javascript",
        src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js",
    )
    soup.head.append(require_script)

    # Add CodeMirror for code editing (loaded after thebe-core)
    codemirror_css = soup.new_tag(
        "link",
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.17/codemirror.min.css",
    )
    soup.head.append(codemirror_css)

    codemirror_theme_css = soup.new_tag(
        "link",
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.17/theme/darcula.min.css",
    )
    soup.head.append(codemirror_theme_css)

    # Wrap code blocks in thebe-compatible structure
    cell_counter = 0
    for pre in soup.find_all("pre"):
        if (
            pre.parent
            and pre.parent.name == "div"
            and "highlight" in pre.parent.get("class", [])
        ):
            cell_id = f"cell-{cell_counter}"
            cell_counter += 1

            # Create wrapper div
            cell_wrapper = soup.new_tag("div")
            cell_wrapper["class"] = "thebe-cell"
            cell_wrapper["data-cell-id"] = cell_id

            # Create source container
            source_container = soup.new_tag("div")
            source_container["class"] = "thebe-source"
            source_container["data-thebe-source"] = ""

            # Create output container
            output_container = soup.new_tag("div")
            output_container["class"] = "thebe-output"

            # Extract code content
            code_content = pre.get_text()

            # Create pre element for source
            source_pre = soup.new_tag("pre")
            source_pre.string = code_content
            source_container.append(source_pre)

            # Add run button
            run_button = soup.new_tag("button")
            run_button["class"] = "cell-run-button"
            svg = soup.new_tag("svg")
            svg["width"] = "24"
            svg["height"] = "24"
            svg["viewBox"] = "0 0 24 24"
            path = soup.new_tag("path")
            path["d"] = "M8 5v14l11-7z"
            path["fill"] = "var(--accent)"
            svg.append(path)
            run_button.append(svg)
            cell_wrapper.append(run_button)

            # Assemble the cell
            cell_wrapper.append(source_container)
            cell_wrapper.append(output_container)

            # Replace original structure
            pre.parent.replace_with(cell_wrapper)

    # Font and base styling
    style_tag = soup.new_tag("style")
    style_tag.string = """
/* Font settings */
pre, code, .thebe-source pre, .cm-editor, .cm-content {
    font-family: 'Fira Code', monospace,'Courier New', monospace !important;
    font-size: 16px;
    line-height: 1.4;
}

body, .notebook, .container {
    font-family: 'Noto Sans', Arial, sans-serif !important;
}

/* Thebe cell structure */
.thebe-cell {
    margin: 1.5rem 0;
    border-radius: 0px;
    overflow: visible;
    position: relative;
}

.thebe-source {
    position: relative;
    margin-bottom: 0;
}

.thebe-editor {
    position: relative;
}

.thebe-source pre {
    margin: 0;
    padding-left: 0.5rem;
    padding-top: 0.5rem;
    padding-bottom: 1rem;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.thebe-output {
    padding-left: 0.5rem;
    min-height: 0;
}

.thebe-output:empty {
    display: none;
}

.thebe-output.thebe-output-has-content {
    display: block;
    border-top: 1px solid var(--text-lite);
}

/* Controls styling */
.thebe-controls {
    position: fixed;
    top: 20px;
    right: 20px;
    display: flex;
    gap: 10px;
    z-index: 1000;
    background: var(--bg);
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.thebe-controls button {
    padding: 0.5rem 1rem;
    border: 1px solid var(--text-lite);
    background: var(--cell-input-bg);
    color: var(--text);
    border-radius: 6px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    transition: all 0.2s;
}

.thebe-controls button:hover {
    background: var(--accent);
    color: var(--color-white);
    border-color: var(--accent);
}

.thebe-controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.thebe-status {
    padding: 0.5rem 1rem;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.thebe-status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ccc;
}

.thebe-status-indicator.connected {
    background: #4caf50;
}

.thebe-status-indicator.connecting {
    background: #ff9800;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* CodeMirror editor styling */
.cm-editor {
    border-radius: 8px 8px 0 0;
}

.cm-scroller {
    overflow-y: auto;
    overflow-x: hidden;
}

.cm-content,
.cm-line {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

.cm-gutters {
    border-radius: 8px 0 0 0;
}

/* Run button for each cell */
.cell-run-button {
    position: absolute;
    left: -3rem;
    width: 38px;
    height: 38px;
    padding: 0;
    background: var(--bg);
    color: black;
    border: 0px ;
    border-radius: 0px;
    cursor: pointer;
    font-size: 14px;
    font-family: 'Inter', sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 1;
    transition: opacity 0.2s;
}

.thebe-cell:hover .cell-run-button {
    opacity: 1;
}

.cell-run-button:hover {
    background: white;
    border: 1px solid var(--accent);
}

/* CodeMirror editor styling */
.thebe-editor {
    position: relative;
}

.thebe-editor .CodeMirror {
}

.thebe-editor .CodeMirror-gutters {
    border-radius: 8px 0 0 0;
}
"""
    soup.head.append(style_tag)

    # Page layout
    page_style = soup.new_tag("style")
    page_style.string = """
body {
    padding-left: 0;
    box-sizing: border-box;
}

.notebook, .container {
    max-width: none;
    margin-left: 0;
    margin-right: auto;
}

@media (max-width: 600px) {
    body {
        padding-left: 12px;
        padding-right: 12px;
    }
}

@media (min-width: 900px) {
    body {
        padding-left: 124px;
        padding-right: 124px;
    }
}
"""
    soup.head.append(page_style)

    # Add anchor links to headings
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if "id" in heading.attrs and not heading.find("a", class_="anchor-link"):
            text_content = heading.get_text()
            anchor = soup.new_tag("a")
            anchor["class"] = "anchor-link"
            anchor["href"] = "#" + heading["id"]
            anchor.string = "#"
            span = soup.new_tag("span")
            span.string = text_content
            heading.contents = [anchor, span]

    # Theme variables
    theme_style = soup.new_tag("style")
    theme_style.string = """
 :root {
     --color-white: #ffffff;
     --bg: #f9f9f6;
     --text: #4D5461;
     --text-lite: #9b9ea7;
     --accent: #0969da;
     --cell-input-bg: color-mix(in oklab, var(--color-white) 60%, transparent);
     --cell-output-bg: transparent;
 }

 .theme-dark {
     --color-white: #ffffff;
     --bg: #1a1a1f;
     --text: #d4d5d9;
     --text-lite: #9b9ca5;
     --accent: #5b8bff;
     --cell-input-bg: color-mix(in oklab, #2a2a35 80%, transparent);
     --cell-output-bg: transparent;
 }

 body {
     background: var(--bg) !important;
     color: var(--text) !important;
     transition: background .2s ease, color .2s ease;
 }

 pre, code {
     background: var(--cell-input-bg) !important;
     color: var(--text) !important;
 }

 .thebe-source pre {
     border: 1px solid var(--text-lite);
     background: var(--cell-input-bg) !important;
     color: var(--text) !important;
 }

 .thebe-output {
     background: var(--cell-output-bg) !important;
     color: var(--text) !important;
 }

 a {
     color: var(--accent);
 }

 .cm-editor {
     color: var(--text) !important;
 }

 .cm-gutters {
     background: var(--cell-input-bg) !important;
     border-right: 1px solid var(--text-lite);
 }

 .jp-RenderedMarkdown {
     color: var(--text) !important;
 }

 .jp-RenderedMarkdown > h1,
 .jp-RenderedMarkdown > h2,
 .jp-RenderedMarkdown > h3,
 .jp-RenderedMarkdown > h4,
 .jp-RenderedMarkdown > h5,
 .jp-RenderedMarkdown > h6 {
     color: var(--accent) !important;
     font-family: 'Lora', serif;
     position: relative;
 }

 .jp-RenderedMarkdown > h1 {
     font-size: 48px;
     font-weight: 500;
     line-height: 1.2;
 }

 .jp-RenderedMarkdown > h2 {
     font-size: 36px;
     font-weight: 500;
     line-height: 1.3;
 }

 .jp-RenderedMarkdown > h3 {
     font-size: 28px;
     font-weight: 500;
     line-height: 1.3;
 }

 .jp-RenderedMarkdown > p, li {
     font-size: 16px;
     line-height: 1.4;
 }

 .jp-Cell-inputArea {
 }

 .jp-InputPrompt {
     position: absolute;
     left: -4rem;
     top: 0.5rem;
     font-family: 'Fira Code', monospace;
     color: var(--text-lite);
     width: 3.5rem;
     text-align: right;
     font-size: 14px;
 }

 .jp-OutputArea-output pre {
     background: var(--cell-output-bg) !important;
     color: var(--text) !important;
     border: none !important;
 }

 .anchor-link {
     opacity: 0;
     transition: opacity 0.2s;
     position: absolute;
     left: -1.5rem;
     top: 0;
     color: var(--text-lite);
     text-decoration: none;
     font-size: 0.8em;
 }

 h1:hover .anchor-link,
 h2:hover .anchor-link,
 h3:hover .anchor-link,
 h4:hover .anchor-link,
 h5:hover .anchor-link,
 h6:hover .anchor-link {
     opacity: 1;
 }

"""
    soup.head.append(theme_style)

    # Theme detector
    theme_script = soup.new_tag("script")
    theme_script.string = """(function(){
    try {
        var stored = localStorage.getItem('site-theme');
        var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
        var prefersDark = mq ? mq.matches : false;
        var useDark = stored ? stored === 'dark' : prefersDark;
        if(useDark) document.documentElement.classList.add('theme-dark');
        else document.documentElement.classList.remove('theme-dark');

        if(!stored && mq) {
            var onChange = function(e){
                document.documentElement.classList.toggle('theme-dark', e.matches);
            };
            if(mq.addEventListener) mq.addEventListener('change', onChange);
            else if(mq.addListener) mq.addListener(onChange);
        }

        window.toggleSiteTheme = function(){
            var cur = document.documentElement.classList.toggle('theme-dark');
            localStorage.setItem('site-theme', cur ? 'dark' : 'light');
            return cur;
        };
    } catch(e){ console.warn('theme init failed', e); }
})();"""
    soup.head.append(theme_script)

    # Add thebe-core integration script
    thebe_script = soup.new_tag("script", type="text/javascript")
    thebe_script.string = """
// Load thebe-core from CDN
(function() {
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/thebe-core@latest/dist/lib/thebe-core.min.js';
    script.onload = function() {
        initializeThebe();
    };
    document.head.appendChild(script);
})();

function initializeThebe() {
    let server = null;
    let session = null;
    let notebook = null;
    let rendermime = null;
    let codeBlocks = [];

    const LOCAL_OPTIONS = {
        serverSettings: {
            baseUrl: 'http://localhost:8888',
            token: 'test-secret',
        },
        kernelOptions: {
            kernelName: 'python3',
        },
    };

    // Create control panel
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'thebe-controls';
    controlsDiv.innerHTML = `
        <div class="thebe-status">
            <div class="thebe-status-indicator" id="status-indicator"></div>
            <span id="status-text">Not connected</span>
        </div>
        <button id="connect-button">Connect</button>
        <button id="run-all-button" disabled>Run All</button>
        <button id="restart-button" disabled>Restart</button>
    `;
    document.body.insertBefore(controlsDiv, document.body.firstChild);

    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const connectButton = document.getElementById('connect-button');
    const runAllButton = document.getElementById('run-all-button');
    const restartButton = document.getElementById('restart-button');

    // Connect button handler
    connectButton.addEventListener('click', async function() {
        try {
            connectButton.disabled = true;
            connectButton.textContent = 'Connecting...';
            statusIndicator.className = 'thebe-status-indicator connecting';
            statusText.textContent = 'Connecting...';

            // Create configuration
            const config = window.thebeCore.api.makeConfiguration(LOCAL_OPTIONS);

            // Connect to server
            server = window.thebeCore.api.connectToJupyter(config);
            await server.ready;

            // Create rendermime registry
            rendermime = window.thebeCore.api.makeRenderMimeRegistry();

            // Start new session
            session = await server.startNewSession(rendermime);

            // Get code blocks from the page
            codeBlocks = [];
            document.querySelectorAll('.thebe-cell').forEach((cell, index) => {
                const sourcePre = cell.querySelector('.thebe-source pre');
                if (sourcePre) {
                    codeBlocks.push({
                        id: `cell-${index}`,
                        source: sourcePre.textContent,
                    });
                }
            });

            // Create notebook from code blocks
            notebook = window.thebeCore.api.setupNotebookFromBlocks(codeBlocks, config, rendermime);

            // Attach session to notebook
            notebook.attachSession(session);

            // Attach cells to DOM
            document.querySelectorAll('.thebe-cell').forEach((cellElement, index) => {
                const cellId = `cell-${index}`;
                const cell = notebook.getCellById(cellId);
                const outputDiv = cellElement.querySelector('.thebe-output');
                if (cell && outputDiv) {
                    cell.attachToDOM(outputDiv);
                }
            });

            statusIndicator.className = 'thebe-status-indicator connected';
            statusText.textContent = 'Connected';
            connectButton.textContent = 'Connected';
            runAllButton.disabled = false;
            restartButton.disabled = false;

            // Setup cells
            setupCells();
        } catch (error) {
            console.error('Connection error:', error);
            statusIndicator.className = 'thebe-status-indicator';
            statusText.textContent = 'Connection failed';
            connectButton.textContent = 'Retry';
            connectButton.disabled = false;
        }
    });

    // Restart button handler
    restartButton.addEventListener('click', async function() {
        if (session) {
            await session.restart();
            statusText.textContent = 'Kernel restarted';

            // Clear all outputs
            document.querySelectorAll('.thebe-output').forEach(output => {
                output.innerHTML = '';
                output.classList.remove('thebe-output-has-content');
            });
        }
    });

    // Run all button handler
    runAllButton.addEventListener('click', function() {
        if (notebook) {
            notebook.executeAll();
        }
    });

    function setupCells() {
        document.querySelectorAll('.thebe-cell').forEach((cell, index) => {
            // Make the pre element editable
            const sourceDiv = cell.querySelector('.thebe-source');
            const preElement = sourceDiv.querySelector('pre');

            if (preElement) {
                preElement.contentEditable = 'true';
                preElement.style.whiteSpace = 'pre-wrap';
                preElement.style.wordWrap = 'break-word';
                preElement.style.fontFamily = 'monospace';
                preElement.style.backgroundColor = 'var(--cell-input-bg)';
                preElement.style.color = 'var(--text)';
                preElement.style.paddingRight = '0rem';
                preElement.style.paddingLeft = '0.5rem';
                preElement.style.paddingTop = '0.5rem';
                preElement.style.paddingBottom = '0.5rem';
                preElement.style.borderRadius = '0px';
                 preElement.style.border = '1px solid var(--text-lite)';
                preElement.style.outline = 'none';
                preElement.style.minHeight = '2rem';

                // Update cell source when content changes
                preElement.addEventListener('input', function() {
                    const cellId = `cell-${index}`;
                    const cell = notebook.getCellById(cellId);
                    if (cell) {
                        cell.source = preElement.textContent;
                    }
                });
            }

            // Add run button event listener
            const runButton = cell.querySelector('.cell-run-button');
            runButton.addEventListener('click', () => executeCell(cell, index));
        });
    }

    async function executeCell(cellElement, index) {
        if (!notebook) return;

        const cellId = `cell-${index}`;

        try {
            // Execute the specific cell (source is already updated via input event)
            await notebook.executeOnly(cellId);
        } catch (error) {
            console.error('Execution error:', error);
        }
    }
}
"""
    soup.body.append(thebe_script)

    # Write modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))


def main():
    os.makedirs("pages", exist_ok=True)

    for file in os.listdir("notebooks"):
        if file.endswith(".ipynb"):
            base_name = file[:-6]
            html_file = f"pages/{base_name}.html"
            notebook_path = f"notebooks/{file}"

            convert_notebook_to_html(notebook_path, "pages", f"{base_name}.html")
            add_thebe_core_to_html(html_file)

            print(f"Converted {file} to {html_file} with Thebe Core integration")


if __name__ == "__main__":
    main()
