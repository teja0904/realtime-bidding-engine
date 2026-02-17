import sys
import time
import logging
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from config import settings
except ImportError:
    st.error("CRITICAL: Could not import project config. Run from project root.")
    st.stop()

def generate_architecture_diagram(save_to_disk=False):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Fix Limits
    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 4)
    
    def draw_node(x, y, label, sublabel, color):
        box = patches.FancyBboxPatch(
            (x, y), 2.5, 1.2, 
            boxstyle="round,pad=0.2", 
            fc=color, ec="white", lw=2
        )
        ax.add_patch(box)
        ax.text(x + 1.25, y + 0.7, label, ha='center', va='center', color='white', weight='bold', fontsize=11)
        ax.text(x + 1.25, y + 0.4, sublabel, ha='center', va='center', color='white', fontsize=9, style='italic')

    draw_node(0, 1.5, "Data Ingestion", "train.txt", "#2c3e50")       
    draw_node(4, 1.5, "Feature Hashing", "Hash & Scale", "#e67e22")      
    draw_node(8, 1.5, "Drift Injection", f"{settings.DRIFT_TYPE}", "#c0392b") 
    draw_node(12, 1.5, "Online Models", "SGD / HAT / FM", "#2980b9")      

    arrow_style = dict(arrowstyle="->", color="#7f8c8d", lw=2, connectionstyle="arc3")
    ax.annotate("", xy=(4, 2.1), xytext=(2.5, 2.1), arrowprops=arrow_style)
    ax.annotate("", xy=(8, 2.1), xytext=(6.5, 2.1), arrowprops=arrow_style)
    ax.annotate("", xy=(12, 2.1), xytext=(10.5, 2.1), arrowprops=arrow_style)

    if save_to_disk:
        output_path = settings.ASSET_DIR / "architecture.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
        return output_path
    else:
        return fig

# --- Main Dashboard UI ---
if __name__ == "__main__":
    st.set_page_config(
        page_title="RTB Monitor",
        layout="wide",
        page_icon="📡",
        initial_sidebar_state="expanded"
    )
    
    alt.themes.enable("dark")

    with st.sidebar:
        st.header("Control Panel")
        
        if st.button("Reset / Clear Logs"):
            log_path = settings.LOG_DIR / "stream_metrics.csv"
            if log_path.exists():
                log_path.unlink()
            st.success("Logs cleared!")
            time.sleep(0.5)
            st.rerun()
            
        st.divider()
        # Keep this checked by default
        auto_refresh = st.checkbox("Auto-Refresh (1s)", value=True)

    st.title(settings.APP_NAME)
    log_file = settings.LOG_DIR / "stream_metrics.csv"
    
    if not log_file.exists():
        st.warning("Waiting for pipeline to start...")
        st.pyplot(generate_architecture_diagram())
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        st.stop()

    try:
        df = pd.read_csv(log_file)
        
        if df.empty:
            st.info("Waiting for data...")
            if auto_refresh:
                time.sleep(1)
                st.rerun()
            st.stop()
            
        latest_event = df['event_id'].max()
        latest_slice = df[df['event_id'] == latest_event]
        
        if not latest_slice.empty:
            leader = latest_slice.sort_values('auc', ascending=False).iloc[0]
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Events Processed", f"{latest_event:,}")
            k2.metric("Drift Status", "ACTIVE" if latest_event > settings.DRIFT_POINT else "Stable")
            k3.metric("Leader Model", leader['model_name'])
            k4.metric("Best AUC", f"{leader['auc']:.4f}")

        st.subheader("Live Model Comparison")
        chart_data = df.iloc[::10, :] if len(df) > 5000 else df
        
        line_chart = alt.Chart(chart_data).mark_line(point=False).encode(
            x=alt.X('event_id', title='Events', axis=alt.Axis(format=',d')),
            y=alt.Y('auc:Q', scale=alt.Scale(domain=[0.5, 0.8]), title='AUC'),
            color='model_name',
            tooltip=['model_name', 'event_id', 'auc:Q', 'log_loss:Q']
        ).properties(height=450).interactive()
        
        drift_rule = alt.Chart(pd.DataFrame({'x': [settings.DRIFT_POINT]})).mark_rule(
            color='red', strokeDash=[4, 4]
        ).encode(x='x')
        
        st.altair_chart(line_chart + drift_rule, use_container_width=True)

        with st.expander("System Architecture", expanded=True):
            st.pyplot(generate_architecture_diagram())

    except Exception as e:
        st.error(f"Dashboard Error: {e}")

    if auto_refresh:
        time.sleep(1)
        st.rerun()