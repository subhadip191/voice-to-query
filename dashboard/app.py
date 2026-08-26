"""
Voice2Query — Interactive Dashboard
====================================
Streamlit-based web dashboard that encapsulates the entire
Speech-to-SQL pipeline with a modern, premium UI.

Usage:
    streamlit run dashboard/app.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Project Root Setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import DB_PATH, WHISPER_MODEL

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice2Query — Speech-to-SQL",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Pipeline step cards */
    .pipeline-step {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .pipeline-step h4 {
        color: #a78bfa;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .pipeline-step .content {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* SQL code display */
    .sql-display {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Fira Code', 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #79c0ff;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    /* Status badges */
    .badge-success {
        display: inline-block;
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-error {
        display: inline-block;
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-info {
        display: inline-block;
        background: rgba(96, 165, 250, 0.15);
        color: #60a5fa;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Custom divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167,139,250,0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "upload_key_counter" not in st.session_state:
    st.session_state.upload_key_counter = 0


# ─── Lazy Module Loading ────────────────────────────────────────────────────
@st.cache_resource
def load_transcriber():
    """Load Whisper model (cached across sessions)."""
    try:
        from modules.asr.transcriber import Transcriber
        return Transcriber(model_name=WHISPER_MODEL)
    except Exception as e:
        st.warning(f"⚠️ ASR module unavailable: {e}")
        return None


@st.cache_resource
def load_generator():
    """Load Text-to-SQL generator."""
    from modules.text_to_sql.generator import TextToSQLGenerator
    return TextToSQLGenerator()


@st.cache_resource
def load_corrector():
    """Load error corrector."""
    from modules.error_correction.corrector import ErrorCorrector
    return ErrorCorrector()


@st.cache_resource
def load_runner():
    """Load query runner."""
    from modules.executor.query_runner import QueryRunner
    return QueryRunner()


def check_database() -> bool:
    """Check if the database exists and is accessible."""
    return Path(DB_PATH).exists()


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎙️ Voice2Query</h1>
    <p>AI-Powered Speech-to-SQL for Interactive Database Exploration</p>
</div>
""", unsafe_allow_html=True)

# ─── Check Database ─────────────────────────────────────────────────────────
if not check_database():
    st.error(
        "⚠️ **Database not found!** "
        "Run `python database/setup_db.py` from the project root to create it."
    )
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Method",
        ["🎤 Record Voice", "📁 Upload Audio", "⌨️ Type Query"],
        index=2,
        help="Choose how to input your database question.",
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("### 📊 Database Info")
    runner = load_runner()

    # Show table counts
    tables_result = runner.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    if tables_result["success"]:
        for _, row in tables_result["data"].iterrows():
            tbl = row["name"]
            count_result = runner.execute(f"SELECT COUNT(*) as cnt FROM {tbl}")
            if count_result["success"]:
                cnt = count_result["data"]["cnt"].iloc[0]
                st.markdown(f"**{tbl}**: {cnt} rows")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown("### 💡 Example Queries")
    examples = [
        "Show all students in Computer Science",
        "What is the average GPA per department?",
        "Top 5 highest paid professors",
        "How many students got an A in each course?",
        "Find students with GPA above average",
        "List scholarships for CS students",
        "Which departments have the most students?",
        "Show courses with more than 3 credits",
    ]
    for ex in examples:
        if st.button(f"📝 {ex}", key=f"ex_{ex}", use_container_width=True):
            st.session_state["example_query"] = ex

# ─── Main Pipeline ──────────────────────────────────────────────────────────

# Step 0: Input
st.markdown("## 🔊 Input Your Question")
transcribed_text = ""

if input_mode == "🎤 Record Voice":
    st.markdown(
        '<span class="badge-info">VOICE INPUT</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Speak your question** — recording stops automatically when you pause.",
    )

    if st.button("🎤 Start Listening", use_container_width=True, type="primary"):
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
        import tempfile
        import threading
        import queue

        SAMPLE_RATE = 16000
        SILENCE_THRESHOLD = 0.02   # Amplitude threshold for silence
        SILENCE_DURATION = 2.5     # Seconds of silence before auto-stop
        MAX_DURATION = 20          # Maximum recording seconds
        BLOCK_SIZE = 1600          # 100ms blocks at 16kHz

        status = st.empty()
        status.info("🔴 **Listening...** Speak now!")

        # Use a queue for thread-safe audio capture
        audio_queue = queue.Queue()
        recording_done = threading.Event()
        all_audio = []
        started_speaking = False
        silent_blocks = 0
        silence_blocks_needed = int(SILENCE_DURATION * SAMPLE_RATE / BLOCK_SIZE)

        def audio_callback(indata, frames, time_info, cb_status):
            """Called by sounddevice for each audio block — no gaps."""
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            ):
                start_time = time.time()

                while (time.time() - start_time) < MAX_DURATION:
                    try:
                        block = audio_queue.get(timeout=0.15)
                    except queue.Empty:
                        continue

                    all_audio.append(block)
                    amplitude = np.abs(block).mean()

                    if amplitude > SILENCE_THRESHOLD:
                        started_speaking = True
                        silent_blocks = 0
                        status.info("🔴 **Listening...** (speaking detected)")
                    elif started_speaking:
                        silent_blocks += 1
                        remaining = max(0, SILENCE_DURATION - silent_blocks * BLOCK_SIZE / SAMPLE_RATE)
                        if remaining > 0:
                            status.info(f"🔴 **Listening...** (silence — stopping in {remaining:.1f}s)")
                        if silent_blocks >= silence_blocks_needed:
                            status.success("✅ Recording complete — processing...")
                            break

            if not started_speaking:
                status.warning("⚠️ No speech detected. Please try again.")
            else:
                # Combine all audio into one continuous array
                audio_data = np.concatenate(all_audio, axis=0)
                tmp_path = tempfile.mktemp(suffix=".wav")
                sf.write(tmp_path, audio_data, SAMPLE_RATE)

                # Playback
                st.audio(tmp_path, format="audio/wav")

                # Transcribe with Whisper
                with st.spinner("🎧 Transcribing with Whisper..."):
                    transcriber = load_transcriber()
                    if transcriber:
                        result = transcriber.transcribe_audio(tmp_path)
                        if result["error"]:
                            st.error(f"❌ {result['error']}")
                        else:
                            transcribed_text = result["text"]
                            st.markdown(
                                f'<div class="pipeline-step">'
                                f'<h4>📝 Step 1 — Transcription</h4>'
                                f'<div class="content">{transcribed_text}</div>'
                                f'<br><span class="badge-success">Confidence: {result["confidence"]:.1%}</span>'
                                f' <span class="badge-info">Language: {result["language"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.error("Whisper model not available. Install: `pip install openai-whisper`")

                # Clean up
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            status.error(f"❌ Microphone error: {e}. Make sure your mic is connected.")

elif input_mode == "📁 Upload Audio":
    st.markdown(
        '<span class="badge-info">FILE UPLOAD</span>',
        unsafe_allow_html=True,
    )

    # Reset button to allow uploading a new file
    col_upl, col_ureset = st.columns([3, 1])
    with col_ureset:
        if st.button("🔄 New Upload", use_container_width=True):
            st.session_state.upload_key_counter += 1
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload an audio file:",
        type=["wav", "mp3", "m4a", "webm", "flac", "ogg"],
        key=f"audio_uploader_{st.session_state.upload_key_counter}",
    )
    if uploaded_file:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        with st.spinner("🎧 Transcribing audio..."):
            transcriber = load_transcriber()
            if transcriber:
                ext = Path(uploaded_file.name).suffix
                result = transcriber.transcribe_from_bytes(
                    uploaded_file.read(), file_extension=ext
                )
                if result["error"]:
                    st.error(f"❌ {result['error']}")
                else:
                    transcribed_text = result["text"]
                    st.markdown(
                        f'<div class="pipeline-step">'
                        f'<h4>📝 Step 1 — Transcription</h4>'
                        f'<div class="content">{transcribed_text}</div>'
                        f'<br><span class="badge-success">Confidence: {result["confidence"]:.1%}</span>'
                        f' <span class="badge-info">Language: {result["language"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error("Whisper model not available. Install: `pip install openai-whisper`")

elif input_mode == "⌨️ Type Query":
    st.markdown(
        '<span class="badge-info">TEXT INPUT</span>',
        unsafe_allow_html=True,
    )

    # Check for example query from sidebar
    default_text = st.session_state.pop("example_query", "")

    transcribed_text = st.text_input(
        "Type your database question in natural language:",
        value=default_text,
        placeholder="e.g., Show me all students in the Computer Science department",
        key="text_input",
    )

# ─── Pipeline Execution ─────────────────────────────────────────────────────
if transcribed_text:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("## ⚡ Pipeline Results")

    # Create columns for pipeline flow
    col1, col2 = st.columns([1, 1])

    with col1:
        # Step 2: Error Correction
        with st.spinner("🔧 Applying error correction..."):
            corrector = load_corrector()
            correction_result = corrector.correct(transcribed_text)
            corrected_text = correction_result["corrected"]

        st.markdown(
            f'<div class="pipeline-step">'
            f'<h4>🔧 Step 2 — Error Correction</h4>'
            f'<div class="content">{corrected_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if correction_result["was_corrected"]:
            with st.expander("🔍 Corrections applied"):
                for c in correction_result["corrections"]:
                    st.markdown(
                        f"- **{c['from']}** → **{c['to']}** "
                        f"_({c['type']}"
                        f"{', sim: ' + str(c.get('similarity', '')) if c.get('similarity') else ''})_"
                    )

        # Step 3: Text-to-SQL
        with st.spinner("🤖 Generating SQL query..."):
            generator = load_generator()
            sql_result = generator.generate_sql(corrected_text)

        if sql_result["error"]:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<h4>🤖 Step 3 — SQL Generation</h4>'
                f'<span class="badge-error">ERROR</span>'
                f'<div class="content" style="color: #f87171;">{sql_result["error"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<h4>🤖 Step 3 — SQL Generation</h4>'
                f'<span class="badge-success">SAFE</span>'
                f'<div class="sql-display">{sql_result["sql"]}</div>'
                f'<br><div class="content" style="font-size:0.85rem; color:#94a3b8;">'
                f'💬 {sql_result["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        if sql_result.get("sql") and sql_result.get("is_safe"):
            # Step 4: Execute Query
            with st.spinner("⚙️ Executing query..."):
                runner = load_runner()
                exec_result = runner.execute(sql_result["sql"])

            if exec_result["success"]:
                st.markdown(
                    f'<div class="pipeline-step">'
                    f'<h4>📊 Step 4 — Results</h4>'
                    f'<span class="badge-success">{exec_result["row_count"]} rows</span> '
                    f'<span class="badge-info">{exec_result["execution_time_ms"]}ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if exec_result["truncated"]:
                    st.warning(f"⚠️ Results truncated to {exec_result['row_count']} rows.")

                # Display data table
                st.dataframe(
                    exec_result["data"],
                    use_container_width=True,
                    hide_index=True,
                )

                # Step 5: Auto-Visualization
                @st.fragment
                def render_visualization(df):
                    if len(df) > 0 and len(df.columns) >= 2:
                        st.markdown(
                            '<div class="pipeline-step">'
                            '<h4>📈 Step 5 — Visualization</h4>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        # Detect chart type based on data
                        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

                        if numeric_cols and text_cols:
                            chart_type = st.selectbox(
                                "Chart Type",
                                ["Bar Chart", "Horizontal Bar", "Line Chart", "Pie Chart"],
                                key="chart_type",
                            )

                            x_col = text_cols[0]
                            y_col = numeric_cols[0]

                            if chart_type == "Bar Chart":
                                fig = px.bar(
                                    df, x=x_col, y=y_col,
                                    color=y_col,
                                    color_continuous_scale="Purp",
                                    template="plotly_dark",
                                )
                            elif chart_type == "Horizontal Bar":
                                fig = px.bar(
                                    df, x=y_col, y=x_col,
                                    orientation="h",
                                    color=y_col,
                                    color_continuous_scale="Purp",
                                    template="plotly_dark",
                                )
                            elif chart_type == "Line Chart":
                                fig = px.line(
                                    df, x=x_col, y=y_col,
                                    markers=True,
                                    template="plotly_dark",
                                )
                                fig.update_traces(line_color="#a78bfa")
                            elif chart_type == "Pie Chart":
                                fig = px.pie(
                                    df, names=x_col, values=y_col,
                                    color_discrete_sequence=px.colors.sequential.Purp,
                                    template="plotly_dark",
                                )

                            fig.update_layout(
                                margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter", color="#e2e8f0"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif numeric_cols and len(df) > 1:
                            # Multiple numeric columns — show as grouped bar
                            fig = go.Figure()
                            for col in numeric_cols[:5]:
                                fig.add_trace(go.Bar(
                                    name=col,
                                    x=df.index.astype(str),
                                    y=df[col],
                                ))
                            fig.update_layout(
                                barmode="group",
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter", color="#e2e8f0"),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Call the visualization fragment
                render_visualization(exec_result["data"])

                # Save to history
                st.session_state.query_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": transcribed_text,
                    "corrected": corrected_text,
                    "sql": sql_result["sql"],
                    "rows": exec_result["row_count"],
                    "time_ms": exec_result["execution_time_ms"],
                })

            else:
                st.markdown(
                    f'<div class="pipeline-step">'
                    f'<h4>📊 Step 4 — Results</h4>'
                    f'<span class="badge-error">EXECUTION ERROR</span>'
                    f'<div class="content" style="color: #f87171;">{exec_result["error"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─── Query History ───────────────────────────────────────────────────────────
if st.session_state.query_history:
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("## 📜 Query History")

    history_df = pd.DataFrame(st.session_state.query_history)
    history_df = history_df[["timestamp", "question", "sql", "rows", "time_ms"]]
    history_df.columns = ["Time", "Question", "SQL", "Rows", "Time (ms)"]

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.rerun()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0 1rem; color: #64748b; font-size: 0.8rem;">
        Voice2Query — Data Management Final Project | Built with Streamlit, Whisper, & Groq Llama 3.3 70B
    </div>
    """,
    unsafe_allow_html=True,
)
