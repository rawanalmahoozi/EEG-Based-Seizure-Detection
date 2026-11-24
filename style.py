
import base64
import os
import streamlit as st


ICON_FILE = "brain_icon.png"


@st.cache_data
def icon_as_text():
    """The icon as text, so it can sit inside the header next to the title."""
    folder = os.path.dirname(os.path.abspath(__file__))
    for path in [ICON_FILE, os.path.join(folder, ICON_FILE)]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


def apply_style():

    css = """
    <style>

    /* ---------------- GLOBAL ---------------- */

    html, body, [data-testid="stAppViewContainer"] {
        background: #F5F8FC !important;
        color: #172B43 !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: #F5F8FC !important;
    }

    [data-testid="stMain"] {
        background: #F5F8FC !important;
    }

    .main .block-container {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
    }

    /* Keep Streamlit's native header and status controls visible. */
    header[data-testid="stHeader"] {
        background: rgba(245, 248, 252, 0.92) !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------------- TYPOGRAPHY ---------------- */

    h1, h2, h3, h4, h5, h6 {
        color: #172B43 !important;
        letter-spacing: -0.025em;
        font-weight: 700;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    p, label, li {
        color: #172B43;
    }

    [data-testid="stCaptionContainer"] {
        color: #667B91 !important;
    }

    .stMarkdown, .stText {
        color: #172B43;
    }

    /* ---------------- CUSTOM HEADER ---------------- */

    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 0.2rem 0 1.1rem 0;
        margin-bottom: 1.4rem;
        border-bottom: 1px solid #DCE6EF;
    }

    .app-logo {
        height: 78px;
        width: auto;
        flex-shrink: 0;
        margin-right: -4px;
    }

    .app-logo-emoji {
        font-size: 3rem;
        line-height: 1;
        flex-shrink: 0;
    }

    .app-name {
        color: #0F1B33;
        font-size: 2.3rem;
        font-weight: 850;
        letter-spacing: 0.01em;
        line-height: 1;
    }

    .app-tagline {
        color: #172B43;
        font-size: 1.02rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* ---------------- HOMEPAGE ---------------- */

    .welcome-label {
        color: #087F9A !important;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .welcome-title {
        color: #172B43 !important;
        font-size: clamp(2rem, 3.2vw, 2.85rem);
        line-height: 1.13;
        font-weight: 780;
        letter-spacing: -0.05em;
        margin: 0 0 0.85rem 0;
    }

    .welcome-description {
        color: #667B91 !important;
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 780px;
        margin-bottom: 1.7rem;
    }

    .upload-heading {
        color: #172B43 !important;
        font-size: 1.25rem;
        font-weight: 720;
        margin-bottom: 0.35rem;
    }

    .upload-description {
        color: #667B91 !important;
        font-size: 0.92rem;
        line-height: 1.65;
        margin-bottom: 1.15rem;
    }

    .workspace-note {
        margin-top: 1.2rem;
        padding: 0.9rem 1rem;
        border-left: 3px solid #087F9A;
        border-radius: 0 9px 9px 0;
        background: #EAF4F8;
        color: #526B82;
        font-size: 0.86rem;
        line-height: 1.6;
    }

    /* ---------------- CONTAINERS / CARDS ---------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE6EF !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 9px rgba(28, 55, 80, 0.035);
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 16px !important;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE6EF !important;
        border-radius: 13px !important;
        padding: 1rem 1.05rem !important;
        box-shadow: 0 2px 8px rgba(28, 55, 80, 0.025);
    }

    [data-testid="stMetricLabel"] {
        color: #667B91 !important;
        font-size: 0.84rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #172B43 !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* ---------------- BUTTONS ---------------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid #DCE6EF !important;
        background: #F0F5FA !important;
        color: #172B43 !important;
        font-weight: 650 !important;
        min-height: 2.65rem;
        transition: all 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #087F9A !important;
        color: #087F9A !important;
        background: #E9F1F7 !important;
    }

    .stButton > button[kind="primary"] {
        background: #087F9A !important;
        border-color: #087F9A !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"]:hover p {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #066A82 !important;
        border-color: #066A82 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:disabled p,
    .stButton > button[kind="primary"]:disabled p {
        color: #667B91 !important;
    }

    .stButton > button:disabled {
        opacity: 0.48 !important;
        color: #667B91 !important;
        background: #E9F1F7 !important;
    }

    /* ---------------- FILE UPLOADER ---------------- */

    [data-testid="stFileUploader"] {
        background: #F8FBFE !important;
        border: 1px dashed #B9CBD9 !important;
        border-radius: 13px !important;
        padding: 0.8rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: #172B43 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #FFFFFF !important;
        border: 1px solid #DCE6EF !important;
        border-radius: 9px !important;
        color: #172B43 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        border-color: #087F9A !important;
        color: #087F9A !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #F8FBFE !important;
        border-color: #B9CBD9 !important;
    }

    /* ---------------- INPUTS ---------------- */

    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #DCE6EF !important;
        color: #172B43 !important;
        border-radius: 9px !important;
    }

    input, textarea {
        color: #172B43 !important;
    }

    [data-baseweb="select"] span {
        color: #172B43 !important;
    }

    /* ---------------- RADIO / CHECKBOX ---------------- */

    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: #172B43 !important;
    }

    [data-testid="stRadio"] p {
        color: #172B43 !important;
    }

    /* ---------------- PROGRESS ---------------- */

    [data-testid="stProgressBar"] > div {
        background: #E7EEF4 !important;
        border-radius: 20px !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #087F9A !important;
        border-radius: 20px !important;
    }

    /* ---------------- ALERTS ---------------- */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid #DCE6EF !important;
    }

    /* ---------------- TABS ---------------- */

    [data-testid="stTabs"] button {
        color: #667B91 !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #087F9A !important;
    }

    /* ---------------- TABLES ---------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE6EF !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ---------------- EXPANDERS ---------------- */

    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE6EF !important;
        border-radius: 12px !important;
    }

    /* ---------------- PENDING REVIEWS (amber, so they are not missed) ---------------- */

    .st-key-pending_card [data-testid="stMetric"] {
        background: #FFF4D6 !important;
        border: 2px solid #E0A100 !important;
        border-left: 8px solid #E0A100 !important;
    }

    .st-key-pending_card [data-testid="stMetricValue"] {
        color: #8A5A00 !important;
        font-size: 2.4rem !important;
    }

    div[class*="st-key-pendingcard"] {
        border: 2px solid #E0A100 !important;
        background: #FFFAEB !important;
        border-radius: 16px !important;
    }

    div[class*="st-key-pendingcard"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* ---------------- STATUS CHIPS ---------------- */

    .status-chip {
        display: inline-flex;
        align-items: center;
        padding: 5px 11px;
        margin: 3px 5px 3px 0;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.3;
        white-space: nowrap;
    }

    .status-red {
        background: rgba(201, 67, 80, 0.10);
        color: #B93645 !important;
        border: 1px solid rgba(201, 67, 80, 0.24);
    }

    .status-green {
        background: rgba(22, 132, 95, 0.10);
        color: #147653 !important;
        border: 1px solid rgba(22, 132, 95, 0.24);
    }

    .status-amber {
        background: rgba(169, 109, 20, 0.10);
        color: #98600F !important;
        border: 1px solid rgba(169, 109, 20, 0.23);
    }

    .status-gray {
        background: #F0F5FA;
        color: #667B91 !important;
        border: 1px solid #DCE6EF;
    }

    .status-teal {
        background: rgba(8, 127, 154, 0.10);
        color: #087F9A !important;
        border: 1px solid rgba(8, 127, 154, 0.23);
    }

    /* ---------------- SECTION DIVIDERS ---------------- */

    .section-separator {
        height: 1px;
        background: #DCE6EF;
        margin: 2rem 0;
    }

    /* ---------------- EEG REVIEW ---------------- */

    #eeg-review-section {
        scroll-margin-top: 1.5rem;
    }

    /* ---------------- SCROLLBAR ---------------- */

    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: #F5F8FC;
    }

    ::-webkit-scrollbar-thumb {
        background: #C9D6E2;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #9EB2C4;
    }

    /* ---------------- RESPONSIVE ---------------- */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .app-name {
            font-size: 1.7rem;
        }

        .app-logo {
            height: 56px;
        }

        .welcome-title {
            font-size: 1.9rem;
        }
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)




def show_banner():

    if st.session_state.get("page", "upload") == "upload":
        return

    icon = icon_as_text()

    if icon:
        logo = '<img class="app-logo" src="data:image/png;base64,' + icon + '" alt="EEG logo">'
    else:
        logo = '<div class="app-logo-emoji">🧠</div>'

    st.markdown(
        '<div class="app-header">'
        + logo
        + '<div>'
        + '<div class="app-name">EEG</div>'
        + '<div class="app-tagline">Seizure Detection and Review Assistant</div>'
        + '</div>'
        + '</div>',
        unsafe_allow_html=True,
    )


def show_welcome_header():

    icon = icon_as_text()

    if icon:
        logo = '<img class="hero-logo" src="data:image/png;base64,' + icon + '" alt="EEG logo">'
    else:
        logo = '<div class="hero-emoji">🧠</div>'

    st.markdown(
        """
        <style>
        .block-container, [data-testid="stMainBlockContainer"] { max-width: 1180px !important; }
        .hero { text-align: center; padding: 1.2rem 0 1.4rem 0; }
        .hero-logo { height: 120px; width: auto; margin-bottom: 0.9rem; }
        .hero-emoji { font-size: 4.5rem; line-height: 1; margin-bottom: 0.9rem; }
        .hero .welcome-description { margin-left: auto; margin-right: auto; margin-bottom: 0.4rem; }
        </style>
        <div class="hero">
        """
        + logo
        + """
            <div class="welcome-label">CLINICAL REVIEW WORKSPACE</div>
            <div class="welcome-title">EEG Analysis Workspace</div>
            <div class="welcome-description">
                Analyse EEG recordings, review events highlighted by the
                automated analysis, and document specialist decisions in
                one organised workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )