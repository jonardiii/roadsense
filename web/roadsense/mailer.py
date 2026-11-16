"""Email helper for RoadSense password reset codes."""
from email.message import EmailMessage
import os
import smtplib

from flask import current_app


def send_reset_code(to_email, code):
    """Send a 6-digit password reset code via SMTP if configured, else log for local dev.

    Returns (sent_via_smtp: bool, dev_code: str | None).
    """
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        current_app.logger.info("Password reset code for %s: %s", to_email, code)
        return False, code

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", "").strip() or user or "noreply@roadsense.local"
    use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() not in ("0", "false", "no")

    msg = EmailMessage()
    msg["Subject"] = "RoadSense Password Reset Code"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        "Hello,\n\n"
        f"Your RoadSense password reset verification code is: {code}\n\n"
        "This code expires in 10 minutes. If you did not request a password reset, "
        "you can ignore this email.\n"
    )

    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if use_tls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return True, None
