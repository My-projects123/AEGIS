OLIVE = "#2F4A32"
OLIVE_DEEP = "#243826"
PANEL = "#FFFFFF"
WHITE = "#FFFFFF"
RED = "#9B2D2D"
TEXT = "#1A2A18"
MUTED = "#4D6350"

def css() -> str:
    return f"""
    <style>
    .stApp {{ background: {OLIVE_DEEP}; color: {WHITE}; }}
    .block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
    h1, h2, h3, h4 {{ font-family: "Barlow Condensed", "Segoe UI", sans-serif; letter-spacing: 0.04em; }}
    .hero-title {{ font-size: 1.7rem; font-weight: 700; color: {WHITE}; margin-bottom: 0.15rem; }}
    .hero-sub {{ color: {WHITE}; font-size: 1.02rem; margin-bottom: 0.4rem; }}
    .ps {{ color: #d5e3d4; font-size: 0.85rem; margin-bottom: 0.8rem; }}
    .card {{ background: {PANEL}; border: 2px solid {OLIVE}; border-radius: 2px; padding: 0.85rem 1rem; color: {TEXT}; }}
    .metric-label {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ color: {OLIVE}; font-size: 1.35rem; font-weight: 700; }}
    .mode-pill {{ display:inline-block; padding: 0.35rem 0.8rem; border-radius: 2px; font-weight: 700; letter-spacing: 0.06em; }}
    .honesty {{ font-size: 0.78rem; color: {WHITE}; background: {OLIVE}; border: 2px solid {WHITE}; padding: 0.6rem 0.8rem; }}
    .stButton>button {{ background: {WHITE}; color: {OLIVE_DEEP}; font-weight: 700; border: 1px solid {WHITE}; }}
    </style>
    """
