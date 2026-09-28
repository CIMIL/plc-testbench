"""Build and locally serve the PLCTestbench documentation site."""

from __future__ import annotations

import argparse
import functools
import shutil
from collections.abc import Sequence
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

DocumentationProfile = Literal["standalone", "embedded"]
DOCUMENTATION_PROFILES: tuple[DocumentationProfile, ...] = ("standalone", "embedded")

_INSTALL_HINT = 'Install documentation support with: pip install "plctestbench[docs]"'


class DocumentationDependencyError(RuntimeError):
    """Raised when documentation support was not installed."""


def _documentation_root() -> Path:
    """Return the packaged MkDocs configuration directory."""
    path = Path(__file__).resolve().with_name("_docs")
    if not path.is_dir():
        raise RuntimeError("The packaged documentation sources could not be found.")
    return path


def _load_mkdocs():
    try:
        from mkdocs.commands.build import build
        from mkdocs.config import load_config
    except ModuleNotFoundError as error:
        raise DocumentationDependencyError(_INSTALL_HINT) from error
    return build, load_config


def _validate_profile(profile: str) -> DocumentationProfile:
    if profile not in DOCUMENTATION_PROFILES:
        choices = ", ".join(DOCUMENTATION_PROFILES)
        raise ValueError(
            f"Unknown documentation profile '{profile}'. Expected one of: {choices}."
        )
    return profile


def _prepare_embedded_content(source_root: Path, destination: Path) -> Path:
    content_directory = destination / "content"
    shutil.copytree(source_root / "content", content_directory)
    shutil.copy2(source_root / "embedded" / "index.md", content_directory / "index.md")
    getting_started_directory = content_directory / "getting-started"
    if not getting_started_directory.is_dir():
        raise RuntimeError(
            "The standalone getting-started documentation could not be found."
        )
    try:
        shutil.rmtree(getting_started_directory)
    except OSError as error:
        raise RuntimeError(
            "Unable to exclude getting-started pages from embedded documentation."
        ) from error
    return content_directory


def build_docs(
    output_dir: str | Path,
    *,
    strict: bool = False,
    profile: DocumentationProfile = "standalone",
) -> Path:
    """Build a static documentation profile into ``output_dir``.

    ``standalone`` contains the package introduction, getting-started pages, and
    repository links. ``embedded`` substitutes its own introduction and excludes
    standalone-only content and repository actions.

    The builder parses package source rather than importing application modules, so
    documentation generation does not initialize database or native integrations.
    """
    selected_profile = _validate_profile(profile)
    destination = Path(output_dir).expanduser().resolve()
    source_root = _documentation_root().resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError(
            "output_dir must not be inside the packaged documentation source"
        )

    build, load_config = _load_mkdocs()
    config_file = source_root / "mkdocs.yml"
    config_overrides: dict[str, str | bool] = {
        "site_dir": str(destination),
        "strict": strict,
    }

    with TemporaryDirectory(prefix="plctestbench-docs-source-") as temporary_directory:
        if selected_profile == "embedded":
            config_file = source_root / "mkdocs.embedded.yml"
            embedded_content = _prepare_embedded_content(
                source_root,
                Path(temporary_directory),
            )
            config_overrides["docs_dir"] = str(embedded_content)

        config = load_config(
            config_file=str(config_file),
            **config_overrides,
        )
        build(config, dirty=False)

    return destination


def _create_server(site_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site_dir))
    return ThreadingHTTPServer((host, port), handler)


def serve_docs(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    output_dir: str | Path | None = None,
    strict: bool = False,
    profile: DocumentationProfile = "standalone",
) -> None:
    """Build one documentation profile and serve it until interrupted.

    For production integrations, use :func:`build_docs` and mount the returned
    static directory with the hosting application's own web framework.
    """
    temporary_output = None
    if output_dir is None:
        temporary_output = TemporaryDirectory(prefix="plctestbench-docs-")
        output_dir = temporary_output.name

    try:
        site_dir = build_docs(output_dir, strict=strict, profile=profile)
        server = _create_server(site_dir, host, port)
        bound_host, bound_port = server.server_address[:2]
        print(
            f"Serving PLCTestbench documentation at http://{bound_host}:{bound_port}/"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        if temporary_output is not None:
            temporary_output.cleanup()


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=DOCUMENTATION_PROFILES,
        default="standalone",
        help="documentation profile to build (default: standalone)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or serve PLCTestbench documentation."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser(
        "build", help="build a static documentation site"
    )
    build_parser.add_argument("output_dir", nargs="?", default="site")
    build_parser.add_argument(
        "--strict", action="store_true", help="fail on MkDocs warnings"
    )
    _add_profile_argument(build_parser)

    serve_parser = commands.add_parser("serve", help="build once and serve locally")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--output", dest="output_dir")
    serve_parser.add_argument(
        "--strict", action="store_true", help="fail on MkDocs warnings"
    )
    _add_profile_argument(serve_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``plctestbench-docs`` command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "build":
        site_dir = build_docs(
            args.output_dir,
            strict=args.strict,
            profile=args.profile,
        )
        print(f"Built PLCTestbench {args.profile} documentation in {site_dir}")
        return 0

    serve_docs(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        strict=args.strict,
        profile=args.profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
