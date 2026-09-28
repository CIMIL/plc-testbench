"""Generate one mkdocstrings page per public PLCTestbench module."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_MODULES = {"__init__", "__version__", "docs"}

modules = []
for source_path in sorted(PACKAGE_ROOT.glob("*.py")):
    module_name = source_path.stem
    if module_name.startswith("_") or module_name in EXCLUDED_MODULES:
        continue
    modules.append(module_name)
    page_path = Path("reference", f"{module_name}.md")
    with mkdocs_gen_files.open(page_path, "w") as page:
        page.write(f"# `plctestbench.{module_name}`\n\n")
        page.write(f"::: plctestbench.{module_name}\n")

with mkdocs_gen_files.open("reference/index.md", "w") as index:
    index.write("# API reference\n\n")
    index.write(
        "This reference is generated from the installed package source and its docstrings. "
        "Use the guides for task-oriented documentation.\n\n"
    )
    for module_name in modules:
        index.write(f"- [`plctestbench.{module_name}`]({module_name}.md)\n")
