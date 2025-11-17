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


def add_thebe_to_html(html_path):
    """Modify the HTML to integrate Thebe for interactive code execution."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove all style tags to avoid conflicts with the main stylesheet
    for style_tag in soup.find_all("style"):
        style_tag.decompose()

    # Add font and preconnect links similar to index.html
    soup.head.append(soup.new_tag("link", rel="preconnect", href="https://fonts.googleapis.com"))
    soup.head.append(soup.new_tag("link", rel="preconnect", href="https://fonts.gstatic.com", crossorigin=True))
    soup.head.append(soup.new_tag(
        "link",
        href=("https://fonts.googleapis.com/css2?family=Great+Vibes&family=IBM+Plex+Mono:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap"),
        rel="stylesheet",
    ))
    soup.head.append(soup.new_tag("link", href="https://fonts.cdnfonts.com/css/sf-pro-display", rel="stylesheet"))

    # Add Monaspace Argon (hosted) for code editing font
    soup.head.append(soup.new_tag("link", href="https://fonts.cdnfonts.com/css/monaspace-argon", rel="stylesheet"))

    # Add CodeMirror and Thebe CSS
    soup.head.append(soup.new_tag("link", rel="stylesheet", href="https://unpkg.com/codemirror@5.65.16/lib/codemirror.css"))
    soup.head.append(soup.new_tag("link", rel="stylesheet", href="https://unpkg.com/thebe@latest/dist/thebe.css"))

    # Add Thebe JS
    soup.head.append(soup.new_tag("script", type="text/javascript", src="https://unpkg.com/thebe@latest/lib/index.js"))

    # Add Thebe config
    # Remove any existing Thebe config script tags to avoid leftover JS functions
    for old_cfg in soup.find_all('script', attrs={'type': 'text/x-thebe-config'}):
        old_cfg.decompose()

    config_script = soup.new_tag('script', type='text/x-thebe-config')
    config_script.string = '''{
    "bootstrap": false,
    "useBinder": false,
    "useJupyterLite": false,
    "requestKernel": true,
    "serverSettings": {
        "baseUrl": "http://localhost:8888",
        "token": "test-secret",
        "appendToken": true
    },
    "kernelOptions": {
        "name": "python",
        "kernelName": "python"
    },
    "mountActivateWidget": true,
    "mountStatusWidget": true,
    "mountRunButton": true,
    "mountRunAllButton": false,
    "mountRestartButton": false,
    "mountRestartAllButton": false,
    "codeMirrorConfig": {
        "theme": "default",
        "lineNumbers": false,
        "readOnly": false,
        "styleActiveLine": false,
        "matchBrackets": false,
        "autoRefresh": false
    }
}'''
    soup.body.append(config_script)

    # Add helper scripts for buttons and run-cell handling
    button_script = soup.new_tag("script")
    button_script.string = '''(function() {
    function adjustPreHeights() {
        try {
            document.querySelectorAll('pre[data-executable]').forEach(function(el) {
                var h = el.getBoundingClientRect().height;
                if (h) el.style.minHeight = h + 'px';
            });
        } catch (e) {
            console.warn('adjustPreHeights error', e);
        }
    }
    // Run now and on DOMContentLoaded
    adjustPreHeights();
    document.addEventListener('DOMContentLoaded', adjustPreHeights);

    var restartBtn = document.getElementById('restart-kernel');
    if (restartBtn) {
        restartBtn.addEventListener('click', function() {
            if (window.thebe && typeof window.thebe.restart === 'function') {
                window.thebe.restart();
            }
        });
    }
    document.addEventListener('click', function(e) {
        var target = e.target || e.srcElement;
        if (target && target.classList && target.classList.contains('run-cell')) {
            if (window.thebe) {
                var pre = target.nextElementSibling;
                if (pre && pre.tagName === 'PRE') {
                    window.thebe.runCell(pre);
                }
            }
        }
    });
})();'''
    soup.body.append(button_script)

    # Mark code blocks as executable by Thebe
    for pre in soup.find_all('pre'):
        if pre.parent and pre.parent.name == 'div' and 'highlight' in pre.parent.get('class', []):
            pre['data-executable'] = 'true'
            pre['data-language'] = 'python'

    # Inject CSS to set the code editor font to Monaspace Argon
    style_tag = soup.new_tag('style')
    style_tag.string = """
pre, code, .thebe pre, .cm-s-default, .CodeMirror, .CodeMirror pre {
    font-family: 'Monaspace Argon', ui-monospace, SFMono-Regular, Menlo, Monaco, 'Roboto Mono', 'Courier New', monospace !important;
    font-size: 14px;
    line-height: 1.4;
}
"""
    soup.head.append(style_tag)

    # Inject page margins and responsive container styles
    page_style = soup.new_tag('style')
    page_style.string = """
/* Left-aligned page layout */
body {
    padding-left: 0;
    padding-right: 24px;
    box-sizing: border-box;
}

/* Keep main content left-aligned and allow it to grow */
.thebe-page-container, .notebook, .container, .body {
    max-width: none;
    margin-left: 0;
    margin-right: auto;
}

/* Small left gutter on narrow viewports, larger right gutter preserved */
@media (max-width: 600px) {
    body {
        padding-left: 12px;
        padding-right: 12px;
    }
}

@media (min-width: 900px) {
    body {
        padding-left: 0;
        padding-right: 64px;
    }
}
"""
    soup.head.append(page_style)

    # Inject dark theme variables and system-theme detector
    theme_style = soup.new_tag('style')
    theme_style.string = """
/* Color variables for light (default) and dark themes */
:root {
    --bg: #ffffff;
    --fg: #111111;
    --muted: #6b6b6b;
    --code-bg: #f5f5f7;
    --link: #0b63d6;
}
.theme-dark, .theme-dark body {
    --bg: #0b0b0f;
    --fg: #e6e6e6;
    --muted: #9b9b9b;
    --code-bg: #0f1113;
    --link: #6ea8ff;
}
body {
    background: var(--bg) !important;
    color: var(--fg) !important;
    transition: background .2s ease, color .2s ease;
}
pre, code, .thebe pre, .CodeMirror, .CodeMirror pre {
    background: var(--code-bg) !important;
    color: var(--fg) !important;
}
a { color: var(--link); }
"""
    soup.head.append(theme_style)

    # Add theme detector + toggle script
    theme_script = soup.new_tag('script')
    theme_script.string = '''(function(){
    try {
        var stored = localStorage.getItem('site-theme');
        var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
        var prefersDark = mq ? mq.matches : false;
        var useDark = stored ? stored === 'dark' : prefersDark;
        if(useDark) document.documentElement.classList.add('theme-dark');
        else document.documentElement.classList.remove('theme-dark');

        // Listen to system changes when no explicit preference stored
        if(!stored && mq) {
            var onChange = function(e){
                document.documentElement.classList.toggle('theme-dark', e.matches);
            };
            if(mq.addEventListener) mq.addEventListener('change', onChange);
            else if(mq.addListener) mq.addListener(onChange);
        }

        // Expose a small toggle function
        window.toggleSiteTheme = function(){
            var cur = document.documentElement.classList.toggle('theme-dark');
            localStorage.setItem('site-theme', cur ? 'dark' : 'light');
            return cur;
        };
    } catch(e){ console.warn('theme init failed', e); }
})();'''
    soup.body.append(theme_script)

    # Add Thebe activate and status widgets and a restart button at the top of the body
    activate_div = soup.new_tag('div', **{"class": 'thebe-activate'})
    status_div = soup.new_tag('div', **{"class": 'thebe-status'})
    restart_button = soup.new_tag('button', id='restart-kernel')
    restart_button.string = 'Restart Kernel'
    soup.body.insert(0, restart_button)
    soup.body.insert(0, status_div)
    soup.body.insert(0, activate_div)

    # Write back the modified HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))


def main():
    # Create pages directory if it doesn't exist
    os.makedirs("pages", exist_ok=True)

    # Process all notebooks in the notebooks folder
    for file in os.listdir("notebooks"):
        if file.endswith(".ipynb"):
            base_name = file[:-6]  # Remove .ipynb extension
            html_file = f"pages/{base_name}.html"
            notebook_path = f"notebooks/{file}"

            # Convert notebook to HTML
            convert_notebook_to_html(notebook_path, "pages", f"{base_name}.html")

            # Add Thebe integration
            add_thebe_to_html(html_file)

            print(f"Converted {file} to {html_file} with Thebe integration")


if __name__ == "__main__":
    main()
