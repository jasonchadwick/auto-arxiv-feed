"""Build and send an email digest of newly-relevant arXiv papers."""

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

    today = datetime.now().strftime("%Y-%m-%d")
    n = len(relevant_papers)
    subject = f"{subject_prefix} {n} new relevant paper{'s' if n != 1 else ''} – {today}"

    text_body = _build_text_body(relevant_papers, today)
    html_body = _build_html_body(relevant_papers, today)

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


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------


def _build_text_body(papers: List[Dict[str, Any]], date: str) -> str:
    n = len(papers)
    lines = [
        f"arXiv Feed Digest – {date}",
        "=" * 50,
        f"Found {n} relevant new paper{'s' if n != 1 else ''}.",
        "",
    ]
    for i, paper in enumerate(papers, start=1):
        title = paper.get("title") or "Unknown title"
        authors = paper.get("authors") or []
        url = paper.get("url") or ""
        max_sim = paper.get("max_similarity", 0.0)
        abstract = paper.get("abstract") or ""
        top_matches = paper.get("top_matches") or []

        lines.append(f"{i}. {title}")
        if authors:
            lines.append(f"   Authors:    {', '.join(authors)}")
        lines.append(f"   URL:        {url}")
        lines.append(f"   Similarity: {max_sim:.4f}")
        if abstract:
            snippet = abstract[:300] + ("…" if len(abstract) > 300 else "")
            lines.append(f"   Abstract:   {snippet}")
        if top_matches:
            match_str = ", ".join(
                f"{pid} ({sim:.3f})" for pid, sim in top_matches[:3]
            )
            lines.append(f"   Similar to: {match_str}")
        lines.append("")
    return "\n".join(lines)


def _build_html_body(papers: List[Dict[str, Any]], date: str) -> str:
    n = len(papers)
    items_html: List[str] = []

    for paper in papers:
        title = paper.get("title") or "Unknown title"
        url = paper.get("url") or "#"
        authors = paper.get("authors") or []
        abstract = paper.get("abstract") or ""
        max_sim = paper.get("max_similarity", 0.0)
        top_matches = paper.get("top_matches") or []

        author_html = (
            f'<p><strong>Authors:</strong> {", ".join(authors)}</p>'
            if authors
            else ""
        )
        snippet = abstract[:400] + ("…" if len(abstract) > 400 else "")
        abstract_html = (
            f"<p><strong>Abstract:</strong> {snippet}</p>" if abstract else ""
        )
        match_html = ""
        if top_matches:
            match_list = ", ".join(
                f"{pid} ({sim:.3f})" for pid, sim in top_matches[:3]
            )
            match_html = f"<p><em>Similar library papers: {match_list}</em></p>"

        items_html.append(
            f"""
            <div style="border-left:3px solid #4a90d9; padding-left:15px; margin-bottom:25px;">
              <h3><a href="{url}">{title}</a></h3>
              {author_html}
              <p><strong>Similarity score:</strong> {max_sim:.4f}</p>
              {abstract_html}
              {match_html}
            </div>
            """
        )

    return f"""
    <html>
    <body>
      <h1>arXiv Feed Digest – {date}</h1>
      <p>Found <strong>{n}</strong> relevant new paper{"s" if n != 1 else ""}.</p>
      {"".join(items_html)}
    </body>
    </html>
    """
