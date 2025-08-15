import os

# Write files
files = {
    os.path.join(root, "app.py"): app_py,
    os.path.join(root, "README.md"): readme_md,
    os.path.join(root, "requirements.txt"): requirements_txt,
    os.path.join(src, "__init__.py"): init_py,
    os.path.join(src, "data_loader.py"): data_loader_py,
    os.path.join(src, "qa_engine.py"): qa_engine_py,
    os.path.join(src, "utils.py"): utils_py,
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
