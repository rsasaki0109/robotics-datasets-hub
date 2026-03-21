"""Streamlit web dashboard for robotics-datasets-hub."""

from __future__ import annotations

import streamlit as st

from rdh.registry import Registry

# -- Color palette for modality tags --
MODALITY_COLORS: dict[str, str] = {
    "vision": "#4CAF50",
    "lidar": "#2196F3",
    "language": "#FF9800",
    "action": "#9C27B0",
    "imu": "#00BCD4",
    "gps": "#F44336",
    "depth": "#3F51B5",
    "audio": "#E91E63",
    "radar": "#795548",
}

DEFAULT_TAG_COLOR = "#607D8B"


def _colored_tag(text: str, color: str) -> str:
    return (
        f'<span style="background-color:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.8em;margin-right:4px;">{text}</span>'
    )


def _link_button(label: str, url: str) -> str:
    if not url:
        return ""
    return (
        f'<a href="{url}" target="_blank" style="text-decoration:none;'
        f'background:#1a1a2e;color:#e0e0e0;padding:3px 10px;border-radius:6px;'
        f'font-size:0.78em;margin-right:4px;border:1px solid #444;">{label}</a>'
    )


@st.cache_resource
def _load_registry() -> Registry:
    return Registry()


def main() -> None:
    st.set_page_config(
        page_title="Robotics Datasets Hub",
        page_icon="🤖",
        layout="wide",
    )

    reg = _load_registry()
    all_entries = reg.all()

    # ── Sidebar filters ──
    st.sidebar.title("Filters")

    search_query = st.sidebar.text_input("Search datasets", placeholder="e.g. SLAM, lidar, driving")

    all_modalities = sorted({m for e in all_entries for m in e.modalities})
    all_tasks = sorted({t for e in all_entries for t in e.tasks})

    selected_modalities = st.sidebar.multiselect("Modalities", all_modalities)
    selected_tasks = st.sidebar.multiselect("Tasks", all_tasks)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**{len(all_entries)}** datasets registered  \n"
        "Install: `pip install robotics-datasets-hub`"
    )

    # ── Apply filters ──
    if search_query:
        entries = reg.search(search_query)
    else:
        entries = list(all_entries)

    if selected_modalities:
        entries = [e for e in entries if set(selected_modalities) & set(e.modalities)]
    if selected_tasks:
        entries = [e for e in entries if set(selected_tasks) & set(e.tasks)]

    # ── Check for detail view via query param ──
    params = st.query_params
    detail_name = params.get("dataset")

    if detail_name:
        _render_detail(reg, detail_name)
        return

    # ── Header ──
    st.title("Robotics Datasets Hub")
    st.markdown(
        "Browse, search, and download curated AI/Robotics datasets with a single command."
    )
    st.markdown("---")

    if not entries:
        st.warning("No datasets match your filters.")
        return

    st.caption(f"Showing {len(entries)} dataset(s)")

    # ── Dataset cards in a grid ──
    cols_per_row = 3
    for row_start in range(0, len(entries), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, entry in enumerate(entries[row_start : row_start + cols_per_row]):
            with cols[col_idx]:
                _render_card(entry)


def _render_card(entry) -> None:
    """Render a single dataset card."""
    modality_tags = " ".join(
        _colored_tag(m, MODALITY_COLORS.get(m, DEFAULT_TAG_COLOR)) for m in entry.modalities
    )
    task_tags = ", ".join(entry.tasks)

    links_html = ""
    if entry.paper_url:
        links_html += _link_button("Paper", entry.paper_url)
    if entry.github_url:
        links_html += _link_button("GitHub", entry.github_url)
    if entry.huggingface_id:
        hf_url = f"https://huggingface.co/datasets/{entry.huggingface_id}"
        links_html += _link_button("HuggingFace", hf_url)
    if entry.project_url:
        links_html += _link_button("Project", entry.project_url)

    description = entry.description
    if len(description) > 120:
        description = description[:117] + "..."

    card_html = f"""
    <div style="border:1px solid #333;border-radius:10px;padding:16px;margin-bottom:12px;
                background:#1a1a2e;min-height:280px;">
        <h4 style="margin:0 0 6px 0;">{entry.display_name}</h4>
        <p style="font-size:0.85em;color:#aaa;margin-bottom:10px;">{description}</p>
        <div style="margin-bottom:8px;">{modality_tags}</div>
        <p style="font-size:0.8em;margin:4px 0;"><b>Tasks:</b> {task_tags}</p>
        <p style="font-size:0.8em;margin:4px 0;"><b>License:</b> {entry.license}</p>
        <p style="font-size:0.8em;margin:4px 0;"><b>Size:</b> {entry.size_hint or "N/A"}</p>
        <div style="margin-top:10px;">{links_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("Download Command"):
        st.code(f"rdh download {entry.name}", language="bash")

    if st.button("Details", key=f"detail_{entry.name}"):
        st.query_params["dataset"] = entry.name
        st.rerun()


def _render_detail(reg: Registry, name: str) -> None:
    """Render the detail view for a single dataset."""
    entry = reg.get(name)

    if st.button("Back to list"):
        st.query_params.clear()
        st.rerun()

    if not entry:
        st.error(f"Dataset '{name}' not found.")
        return

    st.title(entry.display_name)
    st.markdown(f"**`{entry.name}`**")
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Description")
        st.write(entry.description)

        st.subheader("Modalities")
        modality_tags = " ".join(
            _colored_tag(m, MODALITY_COLORS.get(m, DEFAULT_TAG_COLOR)) for m in entry.modalities
        )
        st.markdown(modality_tags, unsafe_allow_html=True)

        st.subheader("Tasks")
        for task in entry.tasks:
            st.markdown(f"- {task}")

        if entry.tags:
            st.subheader("Tags")
            st.write(", ".join(entry.tags))

    with col_right:
        st.subheader("Metadata")
        st.markdown(f"**License:** {entry.license}")
        st.markdown(f"**Size:** {entry.size_hint or 'N/A'}")
        st.markdown(f"**Download method:** {entry.download.get('method', 'wget')}")

        st.subheader("Links")
        if entry.paper_url:
            st.markdown(f"- [Paper]({entry.paper_url})")
        if entry.project_url:
            st.markdown(f"- [Project page]({entry.project_url})")
        if entry.github_url:
            st.markdown(f"- [GitHub]({entry.github_url})")
        if entry.huggingface_id:
            st.markdown(
                f"- [HuggingFace](https://huggingface.co/datasets/{entry.huggingface_id})"
            )

    st.markdown("---")
    st.subheader("Download")
    st.code(f"rdh download {entry.name}", language="bash")

    if entry.download.get("splits"):
        st.markdown("**Available splits:**")
        for split_name, split_pattern in entry.download["splits"].items():
            st.code(f"rdh download {entry.name} --split {split_name}", language="bash")


if __name__ == "__main__":
    main()
