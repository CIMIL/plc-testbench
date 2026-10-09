"""Tests for the opt-in documentation builder."""

from __future__ import annotations

import threading
from urllib.request import urlopen

import pytest

from plctestbench import docs


def test_build_docs_creates_authored_and_generated_pages(tmp_path):
    pytest.importorskip("mkdocs")

    site_dir = docs.build_docs(tmp_path / "site", strict=True)

    assert site_dir == (tmp_path / "site").resolve()
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "getting-started" / "quickstart" / "index.html").is_file()
    assert (site_dir / "reference" / "plc_testbench" / "index.html").is_file()
    assert (site_dir / "search" / "search_index.json").is_file()
    assert (site_dir / "javascripts" / "platform-integration.js").is_file()
    standalone_index = (site_dir / "index.html").read_text(encoding="utf-8")
    assert "javascripts/platform-integration.js" in standalone_index
    assert "github.com/CIMIL/plc-testbench" in standalone_index
    assert (site_dir / "guides" / "human-listening" / "index.html").is_file()
    analyser_html = (site_dir / "reference" / "output_analyser" / "index.html").read_text(encoding="utf-8")
    assert "AWAITING_INPUT" in analyser_html
    assert "trials_per_track" in analyser_html
    assert "loudness_threshold_db" in analyser_html
    assert not (site_dir / "guides" / "study-hub").exists()


def test_embedded_profile_uses_separate_content_and_hides_repository_links(tmp_path):
    pytest.importorskip("mkdocs")

    site_dir = docs.build_docs(tmp_path / "embedded", strict=True, profile="embedded")

    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    search_index = (site_dir / "search" / "search_index.json").read_text(
        encoding="utf-8"
    )
    assert "OpenPLC Studio documentation" in index_html
    guide = site_dir / "guides" / "study-hub" / "index.html"
    assert guide.is_file()
    guide_html = guide.read_text(encoding="utf-8")
    assert "Create my listening link" in guide_html
    assert "Export ratings" in guide_html
    assert "Close collection" in guide_html
    assert "Using the listening Study Hub" in search_index
    assert "guides/study-hub/" in index_html
    # Authored content may link to the core project; the repository action must
    # remain absent from the embedded theme chrome.
    assert "md-source__repository" not in index_html
    assert (site_dir / "guides" / "human-listening" / "index.html").is_file()
    assert not (site_dir / "getting-started").exists()
    assert "Measure packet-loss concealment with confidence" not in search_index
    assert "Installation" not in search_index
    assert (site_dir / "reference" / "plc_testbench" / "index.html").is_file()


def test_build_docs_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown documentation profile"):
        docs.build_docs(tmp_path / "site", profile="invalid")  # type: ignore[arg-type]


def test_build_docs_rejects_output_inside_packaged_sources(tmp_path):
    source_root = docs._documentation_root()

    with pytest.raises(ValueError, match="must not be inside"):
        docs.build_docs(source_root / "site")


def test_create_server_serves_static_files(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text("documentation", encoding="utf-8")
    server = docs._create_server(site_dir, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        with urlopen(f"http://{host}:{port}/", timeout=5) as response:
            assert response.read() == b"documentation"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_docs_dependency_has_install_hint(monkeypatch, tmp_path):
    def unavailable():
        raise docs.DocumentationDependencyError(
            'Install documentation support with: pip install "plctestbench[docs]"'
        )

    monkeypatch.setattr(docs, "_load_mkdocs", unavailable)
    with pytest.raises(
        docs.DocumentationDependencyError, match=r"plctestbench\[docs\]"
    ):
        docs.build_docs(tmp_path / "site")
