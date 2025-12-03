import os
import subprocess
from bs4 import BeautifulSoup
import json


def convert_notebook_to_html(notebook_path, output_dir, output_file):
    """Convert a Jupyter notebook to HTML using nbconvert."""
    subprocess.run(
        [
            "uv",
            "run",
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


def add_thebe_core_to_html(html_path, notebook_path):
    """Modify the HTML to integrate Thebe Core for interactive code execution."""
    # Load notebook to get tags
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)
    
    cell_tags = {}
    for cell in notebook_data.get("cells", []):
        if "id" in cell:
            cell_tags[cell["id"]] = cell.get("metadata", {}).get("tags", [])

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Inject tags into HTML cells
    for cell in soup.find_all("div", class_="jp-Cell"):
        cell_id_attr = cell.get("id", "")
        if cell_id_attr.startswith("cell-id="):
            cell_id = cell_id_attr.replace("cell-id=", "")
            tags = cell_tags.get(cell_id, [])
            for tag in tags:
                cell["class"].append(f"celltag_{tag}")

    # Remove style tags that are NOT inside cell outputs to avoid conflicts
    # We want to preserve inline styles in outputs (e.g., animations)
    for style_tag in soup.find_all("style"):
        # Check if this style tag is inside a cell output
        parent = style_tag.find_parent("div", class_="jp-OutputArea-output")
        if not parent:
            # This is a notebook-level style tag, remove it
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
        href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.17/theme/neo.min.css",
    )
    soup.head.append(codemirror_theme_css)

    # Add CodeMirror JS
    # We need to temporarily hide 'define' to prevent CodeMirror from trying to register with RequireJS
    # which causes "Mismatched anonymous define() module" errors when loaded via script tags.
    hide_define_script = soup.new_tag("script")
    hide_define_script.string = """
        if (window.define) {
            window._define = window.define;
            window.define = null;
        }
    """
    soup.head.append(hide_define_script)

    codemirror_js = soup.new_tag(
        "script",
        type="text/javascript",
        src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.17/codemirror.min.js",
    )
    soup.head.append(codemirror_js)

    codemirror_mode_js = soup.new_tag(
        "script",
        type="text/javascript",
        src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.17/mode/python/python.min.js",
    )
    soup.head.append(codemirror_mode_js)

    restore_define_script = soup.new_tag("script")
    restore_define_script.string = """
        if (window._define) {
            window.define = window._define;
            window._define = null;
        }
    """
    soup.head.append(restore_define_script)

    # Wrap code blocks in thebe-compatible structure
    cell_counter = 0
    # Iterate over code cells specifically
    for cell in soup.find_all("div", class_="jp-CodeCell"):
        # Check for tags
        classes = cell.get("class", [])
        hide_input = "celltag_hide-input" in classes
        show_output = "celltag_show-output" in classes
        output_only = "celltag_output-only" in classes

        # Find the input block (highlight div)
        input_div = cell.find("div", class_="highlight")
        if not input_div:
            continue

        cell_id = f"cell-{cell_counter}"
        cell_counter += 1

        # Create wrapper div
        cell_wrapper = soup.new_tag("div")
        cell_wrapper["class"] = "thebe-cell"
        cell_wrapper["data-cell-id"] = cell_id

        # Create source container
        source_container = soup.new_tag("div")
        source_container["class"] = "thebe-source"
        if hide_input or output_only:
            source_container["class"] += " hidden"
        source_container["data-thebe-source"] = ""

        # Create output container
        output_container = soup.new_tag("div")
        output_container["class"] = "thebe-output"

        # Extract code content
        code_content = input_div.get_text().rstrip()

        # Create pre element for source
        source_pre = soup.new_tag("pre")
        source_pre.string = code_content
        source_container.append(source_pre)

        # Add run button
        if not output_only:
            run_button = soup.new_tag("button")
            run_button["class"] = "cell-run-button"
            svg = soup.new_tag("svg")
            svg["width"] = "24"
            svg["height"] = "24"
            svg["viewBox"] = "0 0 24 24"
            path = soup.new_tag("path")
            path["d"] = "M8 5v14l11-7z"
            path["fill"] = "currentColor"
            svg.append(path)
            run_button.append(svg)
            cell_wrapper.append(run_button)

        # Handle existing outputs
        output_wrapper = cell.find("div", class_="jp-Cell-outputWrapper")
        if output_wrapper:
            # Find the actual output content
            # Usually in jp-OutputArea-output
            outputs = output_wrapper.find_all("div", class_="jp-OutputArea-output")
            for output in outputs:
                # We append the output to our thebe output container
                # We might need to clean it up or just move it
                output_container.append(output)
                output_container["class"] += " thebe-output-has-content"

        # Assemble the cell
        cell_wrapper.append(source_container)
        if hide_input and not output_only:
            placeholder = soup.new_tag("div")
            placeholder["class"] = "code-hidden-placeholder"
            placeholder.string = "Code hidden"
            cell_wrapper.append(placeholder)
        cell_wrapper.append(output_container)

        # Replace the input div with our wrapper
        input_div.replace_with(cell_wrapper)

        # Remove the original output wrapper as we either moved it or want to strip it
        if output_wrapper:
            output_wrapper.decompose()

    # Group side-by-side cells
    all_cells = soup.find_all("div", class_="jp-Cell")
    i = 0
    while i < len(all_cells):
        cell = all_cells[i]
        classes = cell.get("class", [])
        
        # Check for specific side-by-side tags
        width_tag = None
        if "celltag_s50" in classes:
            width_tag = 50
        elif "celltag_s25" in classes:
            width_tag = 25
        elif "celltag_s75" in classes:
            width_tag = 75
            
        if width_tag:
            # We found a start of a side-by-side pair
            if i + 1 >= len(all_cells):
                raise ValueError(f"Cell with s{width_tag} has no following cell.")
            
            next_cell = all_cells[i+1]
            next_classes = next_cell.get("class", [])
            
            next_width_tag = None
            if "celltag_s50" in next_classes:
                next_width_tag = 50
            elif "celltag_s25" in next_classes:
                next_width_tag = 25
            elif "celltag_s75" in next_classes:
                next_width_tag = 75
                
            if not next_width_tag:
                 raise ValueError(f"Cell with s{width_tag} followed by cell without side-side tag.")

            # Validate pairs
            if width_tag == 50 and next_width_tag != 50:
                 raise ValueError(f"Cell with s50 must be followed by s50, found s{next_width_tag}")
            if width_tag == 25 and next_width_tag != 75:
                 raise ValueError(f"Cell with s25 must be followed by s75, found s{next_width_tag}")
            if width_tag == 75 and next_width_tag != 25:
                 raise ValueError(f"Cell with s75 must be followed by s25, found s{next_width_tag}")

            # Create container
            container = soup.new_tag("div")
            container["class"] = "side-by-side-container"
            cell.insert_before(container)
            container.append(cell)
            container.append(next_cell)
            
            i += 2
            continue

        # Fallback for generic side-by-side (optional, keeping for backward compatibility if needed, or remove)
        if "celltag_side-by-side" in classes:
            group = [cell]
            j = i + 1
            while j < len(all_cells):
                next_cell = all_cells[j]
                next_classes = next_cell.get("class", [])
                if "celltag_side-by-side" in next_classes:
                    group.append(next_cell)
                    j += 1
                else:
                    break
            
            if len(group) > 1:
                container = soup.new_tag("div")
                container["class"] = "side-by-side-container"
                group[0].insert_before(container)
                for c in group:
                    container.append(c)
            
            i = j
        else:
            i += 1
    # Font and base styling
    style_tag = soup.new_tag("style")
    style_tag.string = """
/* Font settings */
pre, code, .thebe-source pre, .cm-editor, .cm-content, .CodeMirror {
    font-family: 'Fira Code', monospace,'Courier New', monospace !important;
    font-size: 16px;
    line-height: 1.4;
}

/* Ensure CodeMirror lines have transparent background so selection is visible */
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
    background: transparent !important;
}

body, .notebook, .container {
    font-family: 'Noto Sans', Arial, sans-serif !important;
}

/* Side-by-side layout */
.side-by-side-container {
    display: flex;
    flex-direction: row;
    gap: 2rem;
    width: 100%;
    margin: 1.5rem 0;
}

.side-by-side-container .jp-Cell {
    flex: 1;
    min-width: 0;
    margin: 0 !important;
    align-content: center;
}

.celltag_s50 {
    flex: 0 0 50% !important;
    max-width: 50%;
}

.celltag_s25 {
    flex: 0 0 25% !important;
}

.celltag_s75 {
    flex: 0 0 75% !important;
}

@media (max-width: 768px) {
    .side-by-side-container {
        flex-direction: column;
    }
}
.jp-RenderedHTML{
    margin: 0 !important;
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

.thebe-source.hidden {
    display: none;
}

.thebe-editor {
    position: relative;
}

.thebe-source pre {
    margin: 0;
    padding-left: 2rem;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.45;
}

.thebe-output {
    padding-left: 0.5rem;
    min-height: 0;
    font-family: var(--primary-font) !important;
}

.thebe-output:empty {
    display: none;
}

.thebe-output.thebe-output-has-content {
    display: block;
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
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.thebe-controls button {
    padding: 0.5rem 1rem;
    border: 0px solid var(--text-lite);
    background: var(--cell-input-bg);
    color: var(--text);
    border-radius: 0px;
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

/* Theme toggle button */
.theme-toggle-btn {
    padding: 0.5rem !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    position: relative;
    width: 40px;
    height: 40px;
    border: 0px !important;
    background: transparent !important;
}

.theme-toggle-btn:hover {
    background: var(--accent) !important;
    color: var(--color-white) !important;
}

.theme-toggle-btn svg {
    position: absolute;
    transition: opacity 0.3s ease, transform 0.3s ease;
}

.theme-toggle-btn .sun-icon {
    opacity: 1;
    transform: rotate(0deg);
}

.theme-toggle-btn .moon-icon {
    opacity: 0;
    transform: rotate(180deg);
}

.theme-toggle-btn.dark-mode .sun-icon {
    opacity: 0;
    transform: rotate(180deg);
}

.theme-toggle-btn.dark-mode .moon-icon {
    opacity: 1;
    transform: rotate(0deg);
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
    left: -2.75rem;
    width: 38px;
    height: 38px;
    padding: 0;
    background: transparent;
    color: var(--text-lite);
    border: 0px solid var(--text-lite);
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
    border: 1px solid var(--accent);
    color: var(--accent);
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
.CodeMirror-lines {
    padding: 0px 0 !important;
}
.CodeMirror-vscrollbar{
display: none;
}
.CodeMirror-cursor {
    width: 2px !important;
    background-color: var(--accent) !important;
}

/* CodeMirror 5 selection styling */
.CodeMirror-selected {
    background: #b3d4fc !important;
}

.CodeMirror-focused .CodeMirror-selected {
    background: #b3d4fc !important;
}

.theme-dark .CodeMirror-selected,
.theme-dark .CodeMirror-focused .CodeMirror-selected {
    background: rgba(88, 166, 255, 0.4) !important;
}
"""
    soup.head.append(style_tag)

    # Modal and Placeholder CSS
    modal_style = soup.new_tag("style")
    modal_style.string = """
/* Modal styles */
.modal {
    display: none;
    position: fixed;
    z-index: 2000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.4);
    backdrop-filter: blur(4px);
}

.modal-content {
    background-color: var(--bg);
    margin-top: 1%;
    margin-bottom: 1%;
    margin-left: auto;
    margin-right: auto;
    padding: 3px;
    border: 1px solid var(--text-lite);
    width: 95%;
    max-width: 1400px;
    position: relative;
    max-height: 95vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}

.close {
    position: absolute;
    right: -52px;
    top: 0px;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--bg);
    opacity: 0.5;
    color: var(--text-lite);
    padding: 6px;
    font-size: 28px;
    cursor: pointer;
    align-self: flex-end;
    margin-bottom: 10px;
    line-height: 1;
}

.close:hover,
.close:focus {
    color: var(--text);
    text-decoration: none;
    cursor: pointer;
    opacity: 1;
}

#modal-code {
    margin: 0;
    padding: 1rem;
    overflow: auto;
    background: var(--cell-input-bg);
    border: 1px solid var(--text-lite);
    font-family: 'Fira Code', monospace;
    font-size: 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
    /* Firefox */
    scrollbar-width: thin;
    scrollbar-color: var(--text-lite) var(--bg);
}

/* Custom scrollbar for modal code (WebKit) */
#modal-code::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
#modal-code::-webkit-scrollbar-track {
    background: var(--bg);
}
#modal-code::-webkit-scrollbar-thumb {
    background-color: var(--text-lite);
    border-radius: 4px;
    border: 2px solid var(--bg);
}
#modal-code::-webkit-scrollbar-thumb:hover {
    background-color: var(--text-lite);
    opacity: 0.9;
}

.code-hidden-placeholder {
    background: var(--cell-input-bg);
    border: 1px dashed var(--text-lite);
    padding: 0.75rem;
    text-align: center;
    color: var(--text-lite);
    cursor: pointer;
    margin: 1rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    transition: all 0.2s ease;
}

.code-hidden-placeholder:hover {
    background: var(--bg);
    border-color: var(--accent);
    color: var(--accent);
}
"""
    soup.head.append(modal_style)

    # First Visit Modal CSS
    first_visit_style = soup.new_tag("style")
    first_visit_style.string = """
#first-visit-modal {
    display: none;
    position: fixed;
    z-index: 2100;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.6);
    backdrop-filter: blur(5px);
    align-items: center;
    justify-content: center;
}

#first-visit-modal .modal-content {
    background-color: var(--bg);
    padding: 2rem;
    border: 1px solid var(--text-lite);
    width: 90%;
    max-width: 800px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

#first-visit-modal h2 {
    margin-top: 0;
    color: var(--accent);
    font-family: 'Lora', serif;
    font-size: 24px;
    font-weight: 300;

}

#first-visit-modal p {
    color: var(--text);
    line-height: 1.6;
    font-size: 16px;
    font-family: 'Inter', sans-serif;
}

.command-box {
    background: var(--cell-input-bg);
    border: 1px solid var(--text-lite);
    padding: 1rem;
    position: relative;
    margin-bottom: 1rem;
}

.command-box pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
    font-family: 'Fira Code', monospace;
    font-size: 14px;
    color: var(--text);
    background: transparent !important;
    border: none !important;
}

.copy-btn {
    background: var(--accent);
    border: 1px solid transparent;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
    color: var(--bg);
    transition: all 0.2s;
    margin-left: auto;
    margin-right: 0;
}

.copy-btn:hover {
    color: var(--accent);
    background: var(--bg);
    border-color: var(--accent);
    border-width: 1px;
}

.modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 1rem;
}

.primary-btn {
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    font-weight: 500;
    transition: opacity 0.2s;
}

.primary-btn:hover {
    opacity: 0.9;
}

.github-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-lite);
    text-decoration: none;
    font-size: 14px;
}

.github-link:hover {
    color: var(--accent);
}
"""
    soup.head.append(first_visit_style)

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

main {
    width: 60%;
    margin-left: auto;
    margin-right: auto;
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
     --cell-input-bg: #FEFEFD;
     --cell-output-bg: transparent;
     --primary-font: "Noto Sans", Arial, sans-serif;
     --code-font: "Fira Code", monospace
 }

 .theme-dark {
     --color-white: #ffffff;
     --bg: #141519;
     --text: #ffffff;
     --text-lite: #8b949e;
     --accent: #58a6ff;
     --cell-input-bg: #252931;
     --cell-output-bg: transparent;
 }

.theme-dark .cm-variable {
    color: #05b491;
 }

.theme-dark .cm-number {
    color: #b467d5;
}

.theme-dark .cm-property,  {
    color: #269cf0 !important;
}

.cm-keyword {
    color: #c35d56 !important;
}

.theme-dark .cm-keyword {
    color: #ff7b72 !important;
}

.theme-dark .cm-string {
    color: #fa821a;
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

 .thebe-source > pre {
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

 .jp-RenderedMarkdown {
     width: 75%;
     margin-right: auto;
 }

 .side-by-side-container .jp-RenderedMarkdown {
     width: 100%;
     margin-right: 0;
 }

 .jp-CodeMirrorEditor {
     width: 75%;
     margin-right: auto;
 }

 .side-by-side-container .jp-CodeMirrorEditor {
     width: 100%;
     margin-right: 0;
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

 .mermaid {
  display: none;
 }

 /* Table styling */
 table {
     border-collapse: collapse;
     width: auto;
     margin: 1.5rem auto;
     font-size: 15px;
     background: var(--bg);
     border: 2px solid var(--text-lite);
     overflow: hidden;
 }

 table thead {
     font-weight: 600;
 }

 table thead th {
     padding: 12px 16px;
     text-align: left;
     font-weight: 600;
     color: var(--text);
     border-bottom: 2px solid var(--text-lite);
     font-family: 'Noto Sans', sans-serif;
     border-right: 1px solid var(--text-lite);
 }

 table thead th:last-child {
     border-right: none;
 }

 table tbody tr {
     border-bottom: 1px solid var(--text-lite);
     transition: background-color 0.2s ease;
 }

 table tbody tr:last-child {
     border-bottom: none;
 }

 table tbody tr:hover {
     background: var(--cell-input-bg);
 }

 table tbody td {
     padding: 10px 16px;
     color: var(--text);
     border-right: 1px solid var(--text-lite);
 }

 table tbody td:last-child {
     border-right: none;
 }

 table tbody tr:nth-child(even) {
     background: rgba(0, 0, 0, 0.02);
 }

 .theme-dark table tbody tr:nth-child(even) {
     background: rgba(255, 255, 255, 0.02);
 }

 /* Responsive tables */
 @media screen and (max-width: 768px) {
     table {
         font-size: 14px;
     }
     
     table thead th,
     table tbody td {
         padding: 8px 12px;
     }
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

        // Always listen for system theme changes when no manual preference is stored
        if(mq) {
            var onChange = function(e){
                // Only auto-switch if user hasn't set a manual preference
                if(!localStorage.getItem('site-theme')) {
                    document.documentElement.classList.toggle('theme-dark', e.matches);
                }
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

    // Initialize cells immediately to show syntax highlighting
    setupCells();

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
        <button id="theme-toggle-button" class="theme-toggle-btn" title="Toggle theme">
            <svg class="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
            <svg class="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
        </button>
    `;
    document.body.insertBefore(controlsDiv, document.body.firstChild);

    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const connectButton = document.getElementById('connect-button');
    const runAllButton = document.getElementById('run-all-button');
    const restartButton = document.getElementById('restart-button');
    const themeToggleButton = document.getElementById('theme-toggle-button');

    // Update theme toggle icon based on current theme
    function updateThemeIcon() {
        const isDark = document.documentElement.classList.contains('theme-dark');
        themeToggleButton.classList.toggle('dark-mode', isDark);
    }
    updateThemeIcon();

    // Theme toggle button handler
    themeToggleButton.addEventListener('click', function() {
        window.toggleSiteTheme();
        updateThemeIcon();
    });


    // Connect button handler
    connectButton.addEventListener('click', () => connectToKernel());

    // Auto-connect if session exists
    if (localStorage.getItem('thebeSessionId')) {
        console.log('AlphaToe: Found saved session, auto-connecting...');
        connectToKernel();
    }

    async function connectToKernel() {
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
            console.log('AlphaToe: Server ready', server);

            // Create rendermime registry
            rendermime = window.thebeCore.api.makeRenderMimeRegistry();

            // Try to reconnect to existing session
            const savedSessionId = localStorage.getItem('thebeSessionId');
            let sessionConnected = false;

            if (savedSessionId) {
                try {
                    console.log('AlphaToe: Attempting to reconnect to session:', savedSessionId);
                    
                    // List running sessions to see if ours is still there
                    let runningSessions = [];
                    if (typeof server.listRunningSessions === 'function') {
                         runningSessions = await server.listRunningSessions();
                    } else {
                        console.warn('AlphaToe: server.listRunningSessions is not a function');
                    }
                    
                    console.log('AlphaToe: Running sessions:', runningSessions);

                    const sessionExists = runningSessions.find(s => s.id === savedSessionId);
                    
                    if (sessionExists) {
                        console.log('AlphaToe: Found existing session, connecting...');
                        session = await server.connectToSession(savedSessionId);
                        sessionConnected = true;
                        console.log('AlphaToe: Reconnected to existing session');
                    } else {
                        console.log('AlphaToe: Saved session ID not found in running sessions');
                    }
                } catch (e) {
                    console.warn('AlphaToe: Failed to reconnect to session:', e);
                    localStorage.removeItem('thebeSessionId');
                }
            }

            if (!sessionConnected) {
                console.log('AlphaToe: Starting new session');
                // Start new session
                session = await server.startNewSession(rendermime);
                console.log('AlphaToe: New session started:', session.id);
                // Save session ID
                localStorage.setItem('thebeSessionId', session.id);
            }

            // Get code blocks from the page
            codeBlocks = [];
            document.querySelectorAll('.thebe-cell').forEach((cell, index) => {
                const sourceDiv = cell.querySelector('.thebe-source');
                let sourceCode = '';
                
                // Try to get code from CodeMirror instance if it exists
                if (sourceDiv.cm) {
                    sourceCode = sourceDiv.cm.getValue();
                } else {
                    // Fallback to pre tag content
                    const sourcePre = sourceDiv.querySelector('pre');
                    if (sourcePre) {
                        sourceCode = sourcePre.textContent;
                    }
                }

                if (sourceCode) {
                    codeBlocks.push({
                        id: `cell-${index}`,
                        source: sourceCode,
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
            // setupCells(); // Already set up on load
        } catch (error) {
            console.error('Connection error:', error);
            statusIndicator.className = 'thebe-status-indicator';
            statusText.textContent = 'Connection failed';
            connectButton.textContent = 'Retry';
            connectButton.disabled = false;
            // Clear saved session on fatal error
            localStorage.removeItem('thebeSessionId');
        }
    }

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
            
            // Only initialize CodeMirror if not hidden
            if (!sourceDiv.classList.contains('hidden')) {
                const preElement = sourceDiv.querySelector('pre');

                if (preElement) {
                    const codeContent = preElement.textContent;
                    sourceDiv.innerHTML = ''; // Clear the pre element

                    // Initialize CodeMirror
                    const cm = CodeMirror(sourceDiv, {
                        value: codeContent,
                        mode: "python",
                        theme: "neo",
                        lineNumbers: false,
                        viewportMargin: Infinity,
                        scrollbarStyle: "null",
                        indentUnit: 4,
                        extraKeys: {
                            "Tab": function(cm) {
                                var spaces = Array(cm.getOption("indentUnit") + 1).join(" ");
                                cm.replaceSelection(spaces);
                            }
                        }
                    });
                    
                    // Force selection styling by injecting style element
                    const selectionStyle = document.createElement('style');
                    selectionStyle.textContent = `
                        .CodeMirror-selected {
                            background: rgba(9, 105, 218, 0.35) !important;
                        }
                        .CodeMirror-focused .CodeMirror-selected {
                            background: rgba(9, 105, 218, 0.35) !important;
                        }
                        .CodeMirror-selectionLayer {
                            z-index: 0 !important;
                        }
                    `;
                    document.head.appendChild(selectionStyle);
                    
                    // Store CodeMirror instance on the element for later retrieval
                    sourceDiv.cm = cm;

                    // Style adjustments for CodeMirror to match design
                    cm.getWrapperElement().style.backgroundColor = 'var(--cell-input-bg)';
                    cm.getWrapperElement().style.border = '1px solid var(--text-lite)';
                    cm.getWrapperElement().style.paddingLeft = '1.7rem';
                    cm.getWrapperElement().style.paddingTop = '0.75rem';
                    cm.getWrapperElement().style.paddingBottom = '1rem';
                    cm.getWrapperElement().style.fontFamily = '"Fira Code", monospace';
                    cm.getWrapperElement().style.height = 'auto';
                    
                    // Update cell source when content changes
                    cm.on('change', function() {
                        const cellId = `cell-${index}`;
                        const cell = notebook.getCellById(cellId);
                        if (cell) {
                            cell.source = cm.getValue();
                        }
                    });
                }
            }

            // Add run button event listener
            const runButton = cell.querySelector('.cell-run-button');
            if (runButton) {
                runButton.addEventListener('click', () => executeCell(cell, index));
            }
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

    # Add Modal for hidden code
    modal_html = soup.new_tag("div")
    modal_html["id"] = "code-modal"
    modal_html["class"] = "modal"
    
    modal_content = soup.new_tag("div")
    modal_content["class"] = "modal-content"
    
    close_span = soup.new_tag("span")
    close_span["class"] = "close"
    close_span.string = "×"
    
    modal_pre = soup.new_tag("pre")
    modal_pre["id"] = "modal-code"
    
    modal_content.append(close_span)
    modal_content.append(modal_pre)
    modal_html.append(modal_content)
    soup.body.append(modal_html)

    # Add First Visit Modal HTML
    fv_modal = soup.new_tag("div")
    fv_modal["id"] = "first-visit-modal"
    
    fv_content = soup.new_tag("div")
    fv_content["class"] = "modal-content"
    
    fv_h2 = soup.new_tag("h2")
    fv_h2.string = "Welcome to Alpha Toe!"
    
    fv_p1 = soup.new_tag("p")
    fv_p1.string = "To interact with the code and run simulations, you need to run a local Jupyter server. Please run the following command in your terminal:"
    
    fv_cmd_box = soup.new_tag("div")
    fv_cmd_box["class"] = "command-box"
    
    fv_pre = soup.new_tag("pre")
    fv_pre["id"] = "server-command"
    fv_pre.string = "uvx -p 3.11 jupyter lab --NotebookApp.token=test-secret --NotebookApp.allow_origin='*' --no-browser"
    
    fv_copy_btn = soup.new_tag("button")
    fv_copy_btn["class"] = "copy-btn"
    fv_copy_btn["onclick"] = "copyCommand()"
    fv_copy_btn.string = "Copy"
    
    fv_cmd_box.append(fv_pre)
    
    fv_actions = soup.new_tag("div")
    fv_actions["class"] = "modal-actions"
    
    fv_gh_link = soup.new_tag("a")
    fv_gh_link["href"] = "https://github.com/nottherealsanta/alpha-toe"
    fv_gh_link["class"] = "github-link"
    fv_gh_link["target"] = "_blank"
    fv_gh_link.string = "View on GitHub"
    
    fv_close_btn = soup.new_tag("button")
    fv_close_btn["class"] = "primary-btn"
    fv_close_btn["onclick"] = "closeFirstVisitModal()"
    fv_close_btn.string = "Got it!"
    
    fv_actions.append(fv_gh_link)
    fv_actions.append(fv_close_btn)
    
    fv_content.append(fv_h2)
    fv_content.append(fv_p1)

    fv_cmd_copy = soup.new_tag("div")
    fv_cmd_copy.append(fv_copy_btn)
    fv_cmd_copy.append(fv_cmd_box)

    fv_content.append(fv_cmd_copy)
    fv_content.append(fv_actions)
    
    fv_modal.append(fv_content)
    soup.body.append(fv_modal)

    # Add Modal JS
    modal_script = soup.new_tag("script")
    modal_script.string = """
    document.addEventListener('DOMContentLoaded', function() {
        var modal = document.getElementById("code-modal");
        var modalCode = document.getElementById("modal-code");
        var span = document.getElementsByClassName("close")[0];

        if (span) {
            span.onclick = function() {
                modal.style.display = "none";
            }
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
        
        document.querySelectorAll('.code-hidden-placeholder').forEach(item => {
            item.addEventListener('click', event => {
                var parent = item.parentElement;
                var source = parent.querySelector('.thebe-source pre');
                if (source) {
                    modalCode.textContent = source.textContent;
                    modal.style.display = "block";
                }
            });
        });
    });
    """
    soup.body.append(modal_script)

    # First Visit Modal JS
    fv_script = soup.new_tag("script")
    fv_script.string = """
    function showFirstVisitModal() {
        console.log('AlphaToe: Showing first visit modal function called');
        const modal = document.getElementById('first-visit-modal');
        if (modal) {
            console.log('AlphaToe: Modal element found, setting display to flex');
            modal.style.setProperty('display', 'flex', 'important');
            // Also ensure z-index is high enough
            modal.style.zIndex = '9999';
        } else {
            console.error('AlphaToe: First visit modal element not found!');
        }
    }

    function closeFirstVisitModal() {
        const modal = document.getElementById('first-visit-modal');
        if (modal) modal.style.display = 'none';
        try {
            localStorage.setItem('alphaToeVisited', 'true');
        } catch (e) {
            console.warn('AlphaToe: Failed to set localStorage', e);
        }
    }

    function copyCommand() {
        const command = document.getElementById('server-command').innerText;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(command).then(() => {
                const btn = document.querySelector('#first-visit-modal .copy-btn');
                const originalText = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(() => {
                    btn.innerText = originalText;
                }, 2000);
            }).catch(err => {
                console.error('AlphaToe: Failed to copy text: ', err);
            });
        } else {
            console.warn('AlphaToe: Clipboard API not available');
            alert('Clipboard API not available. Please copy manually.');
        }
    }

    function initFirstVisit() {
        try {
            console.log('AlphaToe: Checking first visit status...');
            const visited = localStorage.getItem('alphaToeVisited');
            console.log('AlphaToe: visited status raw value:', visited);
            
            // Check if visited is NOT 'true' (handles null, 'false', undefined, etc.)
            if (visited !== 'true') {
                console.log('AlphaToe: User has not visited (or not confirmed), showing modal');
                showFirstVisitModal();
            } else {
                console.log('AlphaToe: User has already visited');
            }
        } catch (e) {
            console.error('AlphaToe: Error checking first visit status', e);
            showFirstVisitModal();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initFirstVisit);
    } else {
        initFirstVisit();
    }
    """
    soup.body.append(fv_script)

    # Write modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))


def main():
    os.makedirs("pages", exist_ok=True)

    notebooks_dir = os.path.join("notebooks")
    for file in os.listdir(notebooks_dir):
        if file.endswith(".ipynb"):
            base_name = file[:-6]
            html_file = f"pages/{base_name}.html"
            temp_html_file = f"pages/{base_name}.tmp.html"
            notebook_path = os.path.join(notebooks_dir, file)

            # Generate to temp file
            convert_notebook_to_html(notebook_path, "pages", f"{base_name}.tmp.html")
            add_thebe_core_to_html(temp_html_file, notebook_path)

            # Atomic move
            os.rename(temp_html_file, html_file)

            print(f"Converted {file} to {html_file} with Thebe Core integration")


if __name__ == "__main__":
    main()
