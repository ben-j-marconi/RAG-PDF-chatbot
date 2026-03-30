"""
ingest.py — CLI ingestion script.

Run before launching the Streamlit app to index your PDF documents.

Usage:
    python ingest.py                # Ingest new/changed documents only
    python ingest.py --force        # Re-index all documents
    python ingest.py --status       # Show registry status without ingesting
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.registry import DocumentRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG knowledge base."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-index all documents even if unchanged."
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print registry status and exit without ingesting."
    )
    args = parser.parse_args()

    config = get_config()

    if args.status:
        _print_status(config)
        return

    print("=" * 60)
    print("  Property Valuation RAG — Document Ingestion Pipeline")
    print("=" * 60)
    print(f"  PDF directory : {config.pdf_dir}")
    print(f"  Registry      : {config.registry_path}")
    print(f"  Vector store  : {config.chroma_dir}")
    print(f"  Embedding     : {config.embedding_model}")
    print(f"  Reranker      : {config.reranker_model}")
    print("=" * 60)

    if args.force:
        print("  Mode: FORCE REINDEX (all documents will be reprocessed)")
    else:
        print("  Mode: DELTA (new/changed documents only)")
    print()

    pipeline = IngestionPipeline(
        config=config,
        progress_callback=lambda msg: print(f"  {msg}"),
    )

    results = pipeline.run(force_reindex=args.force)

    print()
    print("=" * 60)
    print(f"  ✓ Processed : {results['processed']} document(s)")
    print(f"  → Skipped   : {results['skipped']} document(s) (unchanged)")
    print(f"  ✗ Failed    : {results['failed']} document(s)")
    print("=" * 60)
    print()

    if results["failed"] > 0:
        print("  Some documents failed. Check logs for details.")
        sys.exit(1)

    print("  Ready! Start the app with: streamlit run app.py")
    print()


def _print_status(config):
    """Print current registry and index status."""
    print("\n  Registry Status")
    print("  " + "-" * 40)
    try:
        registry = DocumentRegistry(config.registry_path)
        summary = registry.summary()
        for key, val in summary.items():
            print(f"  {key:20s}: {val}")

        print()
        print("  Documents:")
        for rec in registry.all_records():
            status_icon = {"indexed": "✓", "failed": "✗", "pending": "…"}.get(
                rec["status"], "?"
            )
            print(f"    {status_icon} {rec['filename']:50s} v{rec['version']} "
                  f"[{rec['status']}] {rec.get('chunk_count', 0)} chunks")
    except FileNotFoundError:
        print("  Registry not found. Run ingestion first.")
    print()


if __name__ == "__main__":
    main()
