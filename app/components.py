"""
UI Components and visualization helpers for the BankingLLM-Optimizer Streamlit application.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


def render_metric_card(title: str, value: str, delta: Optional[str] = None):
    """Displays a styled metric card."""
    st.metric(label=title, value=value, delta=delta)


def render_prediction_badge(intent: str, confidence: float, needs_review: bool):
    """Displays prominent prediction badges with confidence color coding."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Predicted Intent")
        st.info(f"**`{intent}`**")
    with col2:
        st.markdown("##### Confidence")
        if confidence >= 0.80:
            st.success(f"**{confidence:.1%}** (High)")
        elif confidence >= 0.60:
            st.warning(f"**{confidence:.1%}** (Moderate)")
        else:
            st.error(f"**{confidence:.1%}** (Low)")
    with col3:
        st.markdown("##### Routing Status")
        if needs_review:
            st.warning("?? **Requires Human Review**")
        else:
            st.success("? **Automated (Accepted)**")


def render_retrieved_examples(examples: List[Dict[str, Any]]):
    """Renders dynamically retrieved demonstration cards with similarity scores."""
    if not examples:
        st.caption("No demonstrations retrieved (Zero-Shot / Static strategy).")
        return

    st.markdown(f"#### Retrieved Demonstrations ({len(examples)} exemplars from Training Pool)")
    for i, ex in enumerate(examples, 1):
        sim = ex.get("similarity", 0.0)
        cat = ex.get("category", "unknown")
        text = ex.get("text", "")
        with st.expander(f"Exemplar {i}: **{cat}** (Cosine Similarity: `{sim:.3f}`)", expanded=(i == 1)):
            st.markdown(f"> \"{text}\"")
            st.caption(f"Intent Label: **{cat}** | Index: {ex.get('pool_index', 'N/A')}")


def render_benchmark_charts(summary_df: pd.DataFrame):
    """Renders comparison bar charts for accuracy, F1, latency, and cost."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Macro-F1 / Accuracy
    axes[0].bar(summary_df["Strategy"], summary_df["Macro-F1"], color="#1f77b4", alpha=0.85)
    axes[0].set_title("Macro-F1 by Strategy", fontsize=12)
    axes[0].set_ylabel("Macro-F1 Score")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Latency (P95)
    axes[1].bar(summary_df["Strategy"], summary_df["P95 Latency (ms)"], color="#ff7f0e", alpha=0.85)
    axes[1].set_title("P95 Latency (ms)", fontsize=12)
    axes[1].set_ylabel("Latency in ms")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Estimated Cost per 1K Queries
    axes[2].bar(summary_df["Strategy"], summary_df["Cost / 1K ($)"], color="#2ca02c", alpha=0.85)
    axes[2].set_title("Cost per 1,000 Queries (USD)", fontsize=12)
    axes[2].set_ylabel("Cost ($)")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_automated_reply(reply_text: str, intent: str, confidence: float, needs_review: bool):
    """Renders the AI-generated automated customer support response."""
    st.markdown("#### 💬 Automated Customer Response")
    if needs_review:
        st.warning("⚠️ **Inquiry Escalated**: This request was flagged for human specialist review. The message below is the drafted acknowledgement for the customer.")
    else:
        st.success(f"🤖 **Automated Resolution** (Intent: `{intent}`, Confidence: `{confidence:.1%}`)")

    st.markdown(
        f"""
        <div style="background-color: rgba(59, 130, 246, 0.08); border-left: 4px solid #3B82F6; padding: 16px; border-radius: 6px; margin-bottom: 12px; font-size: 1.05rem; line-height: 1.6;">
            {reply_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.button("📋 Copy Reply", key="btn_copy_reply", use_container_width=True)
    with col2:
        st.button("✏️ Edit Draft", key="btn_edit_draft", use_container_width=True)
    with col3:
        if st.button("✉️ Dispatch Reply to Customer", type="primary", key="btn_send_reply", use_container_width=True):
            st.toast("✅ Response dispatched to customer!", icon="📨")
