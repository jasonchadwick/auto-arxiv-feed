"""Build and send an email digest of newly-relevant arXiv papers."""

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def send_digest(
    relevant_papers: List[Dict[str, Any]],
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_address: str,
    to_address: str,
    subject_prefix: str = "[arXiv Feed]",
    dry_run: bool = False,
) -> bool:
    """Send (or print, in dry-run mode) an email digest.

    Args:
        relevant_papers: List of paper dicts (see :func:`_build_text_body` for keys).
        smtp_host: Outgoing SMTP server hostname.
        smtp_port: SMTP port (usually 587 for STARTTLS or 465 for SSL).
        smtp_user: SMTP username.
        smtp_password: SMTP password.
        from_address: Sender address shown in the email.
        to_address: Recipient address.
        subject_prefix: String prepended to the email subject.
        dry_run: When ``True``, print the email to stdout instead of sending.

    Returns:
        ``True`` on success (or dry-run), ``False`` on send failure.
    """
    if not relevant_papers:
        logger.info("No relevant papers – nothing to send.")
        return True

    # Keep digest ordering deterministic and relevance-first.
    sorted_papers = sorted(
        relevant_papers,
        key=lambda p: float(p.get("max_similarity", 0.0) or 0.0),
        reverse=True,
    )

    today = datetime.now().strftime("%Y-%m-%d")
    n = len(sorted_papers)
    subject = f"{subject_prefix} {n} new relevant paper{'s' if n != 1 else ''} – {today}"

    text_body = _build_text_body(sorted_papers, today)
    html_body = _build_html_body(sorted_papers, today)

    if dry_run:
        separator = "=" * 60
        print(f"\n{separator}")
        print("DRY RUN – email that would be sent:")
        print(f"  From:    {from_address}")
        print(f"  To:      {to_address}")
        print(f"  Subject: {subject}")
        print(separator)
        print(text_body)
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_address, [to_address], msg.as_string())
        logger.info("Digest sent to %s (%d paper(s)).", to_address, n)
        return True
    except smtplib.SMTPException as exc:
        logger.error("Failed to send email: %s", exc)
        return False


def send_error_notification(
    script_name: str,
    error_message: str,
    traceback_text: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_address: str,
    to_address: str,
    subject_prefix: str = "[arXiv Feed]",
    dry_run: bool = False,
) -> bool:
    """Send an email notifying that a script failed with an unhandled error."""
    if not smtp_host:
        logger.warning(
            "Cannot send error notification for %s: SMTP host is not configured.",
            script_name,
        )
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"{subject_prefix} ERROR in {script_name} - {today}"

    text_body = (
        f"auto-arxiv-feed error notification\n"
        f"================================\n"
        f"Script: {script_name}\n"
        f"Time:   {datetime.now().isoformat()}\n\n"
        f"Error:\n{error_message}\n\n"
        f"Traceback:\n{traceback_text}\n"
    )
    html_body = (
        "<html><body>"
        f"<h2>auto-arxiv-feed error notification</h2>"
        f"<p><strong>Script:</strong> {escape(script_name)}</p>"
        f"<p><strong>Time:</strong> {escape(datetime.now().isoformat())}</p>"
        f"<p><strong>Error:</strong> {escape(error_message)}</p>"
        f"<h3>Traceback</h3><pre>{escape(traceback_text)}</pre>"
        "</body></html>"
    )

    if dry_run:
        separator = "=" * 60
        print(f"\n{separator}")
        print("DRY RUN - error email that would be sent:")
        print(f"  From:    {from_address}")
        print(f"  To:      {to_address}")
        print(f"  Subject: {subject}")
        print(separator)
        print(text_body)
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_address, [to_address], msg.as_string())
        logger.info("Error notification sent to %s.", to_address)
        return True
    except smtplib.SMTPException as exc:
        logger.error("Failed to send error notification email: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------


def _build_text_body(papers: List[Dict[str, Any]], date: str) -> str:
    n = len(papers)
    cluster_matched = [p for p in papers if not p.get("override_by_terms")]
    keyword_overrides = [p for p in papers if p.get("override_by_terms")]

    lines = [
        f"arXiv Feed Digest – {date}",
        "=" * 50,
        f"Found {n} relevant new paper{'s' if n != 1 else ''}.",
        "",
    ]

    if cluster_matched:
        lines.append("Cluster-matched items")
        lines.append("-" * 50)
        for i, paper in enumerate(cluster_matched, start=1):
            title = paper.get("title") or "Unknown title"
            authors = paper.get("authors") or []
            url = paper.get("url") or ""
            max_sim = paper.get("max_similarity", 0.0)
            abstract = paper.get("abstract") or ""

            lines.append(f"{i}. {title}")
            if authors:
                lines.append(f"   Authors:    {', '.join(authors)}")
            lines.append(f"   URL:        {url}")
            lines.append(f"   Similarity: {max_sim:.4f}")
            if abstract:
                snippet = abstract[:300] + ("…" if len(abstract) > 300 else "")
                lines.append(f"   Abstract:   {snippet}")
            closest_col = paper.get("closest_collection") or ""
            col_score = paper.get("closest_collection_score", 0.0)
            if closest_col:
                lines.append(f"   Collection: {closest_col} ({col_score:.3f})")
            lines.append("")

    if keyword_overrides:
        lines.append("Items not in clusters but overridden by keyword")
        lines.append("-" * 50)
        lines.append(
            "These matched always-include keywords but were below the LOF threshold."
        )
        lines.append("")
        for i, paper in enumerate(keyword_overrides, start=1):
            title = paper.get("title") or "Unknown title"
            authors = paper.get("authors") or []
            url = paper.get("url") or ""
            max_sim = paper.get("max_similarity", 0.0)
            abstract = paper.get("abstract") or ""
            override_terms = paper.get("override_terms") or []

            lines.append(f"{i}. {title}")
            if authors:
                lines.append(f"   Authors:    {', '.join(authors)}")
            lines.append(f"   URL:        {url}")
            lines.append(f"   Similarity: {max_sim:.4f}")
            if override_terms:
                lines.append(f"   Override:   keyword(s): {', '.join(override_terms)}")
            if abstract:
                snippet = abstract[:300] + ("…" if len(abstract) > 300 else "")
                lines.append(f"   Abstract:   {snippet}")
            closest_col = paper.get("closest_collection") or ""
            col_score = paper.get("closest_collection_score", 0.0)
            if closest_col:
                lines.append(f"   Collection: {closest_col} ({col_score:.3f})")
            lines.append("")

    return "\n".join(lines)


def _render_html_cards(items: List[Dict[str, Any]], start_idx: int = 1) -> List[str]:
    cards: List[str] = []
    for idx, paper in enumerate(items, start=start_idx):
        title = escape(paper.get("title") or "Unknown title")
        url = escape(paper.get("url") or "#")
        authors = paper.get("authors") or []
        abstract = paper.get("abstract") or ""
        max_sim = paper.get("max_similarity", 0.0)
        override_terms = paper.get("override_terms") or []

        author_text = escape(", ".join(authors)) if authors else ""
        author_html = (
            f'<p class="meta-row"><strong>Authors:</strong> {author_text}</p>'
            if authors
            else ""
        )

        snippet = abstract[:400] + ("..." if len(abstract) > 400 else "")
        snippet = escape(snippet)
        abstract_html = (
            f'<p class="abstract"><strong>Abstract:</strong> {snippet}</p>'
            if abstract
            else ""
        )

        closest_col = escape(paper.get("closest_collection") or "")
        col_score = paper.get("closest_collection_score", 0.0)
        collection_html = (
            f'<p class="collection-path">'
            f'<strong>Closest collection:</strong> '
            f'<span class="collection-crumb">{closest_col}</span>'
            f'<span class="collection-score">({col_score:.3f})</span>'
            f'</p>'
            if closest_col
            else ""
        )

        override_html = ""
        if paper.get("override_by_terms") and override_terms:
            override_html = (
                f'<p class="override-note"><strong>Override:</strong> '
                f'keyword(s): {escape(", ".join(override_terms))}</p>'
            )

        cards.append(
            f"""
            <article class="paper-card">
              <div class="rank-badge">#{idx}</div>
              <h3><a href="{url}">{title}</a></h3>
              {author_html}
              <p class="score-row"><strong>Relevance score:</strong> <span class="score">{max_sim:.4f}</span></p>
              {override_html}
              {collection_html}
              {abstract_html}
            </article>
            """
        )
    return cards


def _build_html_body(papers: List[Dict[str, Any]], date: str) -> str:
    n = len(papers)
    cluster_matched = [p for p in papers if not p.get("override_by_terms")]
    keyword_overrides = [p for p in papers if p.get("override_by_terms")]

    main_html = _render_html_cards(cluster_matched, start_idx=1)
    override_html: List[str] = []
    if keyword_overrides:
        override_html.append(
            """
            <section class="section-header">
              <h2>Items not in clusters but overridden by keyword</h2>
              <p>These matched always-include keywords but scored below the LOF threshold.</p>
            </section>
            """
        )
        override_html.extend(_render_html_cards(keyword_overrides, start_idx=1))

    return f"""
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: #eef3f8;
                    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
                    color: #1f2937;
                }}
                .container {{
                    max-width: 860px;
                    margin: 0 auto;
                    padding: 24px 16px 36px;
                }}
                .header {{
                    background: linear-gradient(135deg, #0f3a5f, #1f6ea9);
                    color: #ffffff;
                    border-radius: 14px;
                    padding: 20px 22px;
                    margin-bottom: 20px;
                }}
                .header h1 {{
                    margin: 0 0 8px;
                    font-size: 24px;
                    line-height: 1.2;
                }}
                .header p {{
                    margin: 0;
                    opacity: 0.95;
                }}
                .paper-card {{
                    position: relative;
                    background: #ffffff;
                    border: 1px solid #d8e2ec;
                    border-radius: 12px;
                    padding: 16px 16px 14px;
                    margin-bottom: 14px;
                    box-shadow: 0 2px 8px rgba(15, 58, 95, 0.08);
                }}
                .rank-badge {{
                    position: absolute;
                    top: 10px;
                    right: 12px;
                    background: #e6f1fb;
                    color: #0f4f83;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                    padding: 3px 9px;
                }}
                h3 {{
                    margin: 0 32px 8px 0;
                    font-size: 18px;
                    line-height: 1.3;
                }}
                a {{
                    color: #0f4f83;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                .meta-row, .score-row, .abstract {{
                    margin: 8px 0;
                    font-size: 14px;
                    line-height: 1.5;
                }}
                .score {{
                    color: #0a6a45;
                    font-weight: 700;
                }}
                .collection-path {{
                    margin: 6px 0;
                    font-size: 13px;
                }}
                .collection-crumb {{
                    color: #4b3f82;
                    font-weight: 600;
                    font-family: monospace;
                    background: #f0edf9;
                    border-radius: 4px;
                    padding: 1px 5px;
                    margin-right: 4px;
                }}
                .collection-score {{
                    color: #7c6fbb;
                    font-size: 12px;
                }}
                .section-header {{
                    background: #f3f8ff;
                    border: 1px solid #d8e2ec;
                    border-radius: 10px;
                    padding: 10px 14px;
                    margin: 20px 0 12px;
                }}
                .section-header h2 {{
                    margin: 0 0 4px;
                    font-size: 17px;
                    color: #153a60;
                }}
                .section-header p {{
                    margin: 0;
                    font-size: 13px;
                    color: #334155;
                }}
                .override-note {{
                    margin: 6px 0;
                    font-size: 13px;
                    color: #7a2600;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <section class="header">
                    <h1>arXiv Feed Digest - {date}</h1>
                    <p>Found <strong>{n}</strong> relevant new paper{'s' if n != 1 else ''}, sorted by relevance score.</p>
                </section>
                {''.join(main_html)}
                {''.join(override_html)}
            </div>
        </body>
        </html>
    """
