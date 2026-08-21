import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json

st.set_page_config(page_title="10-K Network Explorer", layout="wide", page_icon="🔍")

st.title("🔍 10-K Network Explorer")
st.markdown("""
**Interactive visualization of company business descriptions and segment relationships**
Explore how companies position themselves in embedding space based on their 10-K Item 1 filings.
""")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("10k_network_panel_100companies_5years.csv")
    df["Segment_List"] = df["Segment"].apply(json.loads)
    df["Involvements"] = df["Segment_Involvements"].apply(json.loads)
    # Products_Services is always a subset of Segment_List in this dataset -- it's the
    # company's *core / headline* product lines, vs. Segment_List which is every declared
    # segment. That distinction is what lets us simulate "drop one product line" separately
    # from "drop an entire segment." Item_1_Snippet, by contrast, is templated boilerplate
    # that just restates Segment_List in a sentence -- it carries no independent signal, so
    # it isn't used as a similarity source here.
    df["Core_Products"] = df["Products_Services"].apply(lambda s: [x.strip() for x in s.split(",")])
    return df

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ Filters")

year = st.sidebar.slider("Select Year", 2021, 2025, 2025)
max_companies = st.sidebar.slider("Number of Companies to Display", 10, 100, 50)

min_mcap, max_mcap = st.sidebar.slider(
    "Market Cap Range ($B)",
    float(df["Market_Cap_B"].min()),
    float(df["Market_Cap_B"].max()),
    (float(df["Market_Cap_B"].min()), float(df["Market_Cap_B"].max())),
)

all_sectors = sorted(df["Sector"].unique().tolist())
selected_sectors = st.sidebar.multiselect("Select Sectors", all_sectors, default=all_sectors)

all_segments = sorted({s for segs in df["Segment_List"] for s in segs})
selected_segments = st.sidebar.multiselect("Select Segments", all_segments, default=all_segments)

edge_threshold = st.sidebar.slider("Edge Distance Threshold", 2.0, 10.0, 6.0)

show_segment_nodes = st.sidebar.checkbox("Show segment nodes", value=True)
show_labels = st.sidebar.checkbox("Show company labels on chart", value=False)
layout_mode = st.sidebar.radio("Layout", ["Embedding space", "Force-directed (spring)"], index=0)

# ---------------------------------------------------------------------------
# Filter data
# ---------------------------------------------------------------------------
df_year = df[df["Year"] == year].copy()
df_year = df_year[df_year["Sector"].isin(selected_sectors)]
df_year = df_year[(df_year["Market_Cap_B"] >= min_mcap) & (df_year["Market_Cap_B"] <= max_mcap)]
df_year = df_year[df_year["Segment_List"].apply(lambda x: any(s in selected_segments for s in x))]
df_year = df_year.sort_values("Market_Cap_B", ascending=False).head(max_companies)

# ---------------------------------------------------------------------------
# Empty-state guard
# ---------------------------------------------------------------------------
if df_year.empty:
    st.warning(
        "No companies match the current filters. Try widening the market cap range, "
        "selecting more sectors/segments, or increasing the company count."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Stats row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies", len(df_year))
col2.metric("Avg Market Cap", f"${df_year['Market_Cap_B'].mean():.1f}B")
col3.metric("Sectors", df_year["Sector"].nunique())
col4.metric("Avg Segments", f"{df_year['Num_Segments'].mean():.1f}")

# ---------------------------------------------------------------------------
# Sector color palette (consistent across reruns, up to ~24 sectors)
# ---------------------------------------------------------------------------
palette = px.colors.qualitative.Alphabet
sector_colors = {sec: palette[i % len(palette)] for i, sec in enumerate(all_sectors)}

# ---------------------------------------------------------------------------
# Build graph (cached on the filtered frame + threshold)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_graph(_frame: pd.DataFrame, cache_key: tuple, threshold: float, include_segments: bool):
    # `_frame` is excluded from Streamlit's hashing (leading underscore) because it has
    # list-valued columns (Segment_List etc.) that aren't hashable. `cache_key` -- a plain
    # tuple of company/year pairs -- stands in as the hashable signature of which rows
    # are actually in play, so the cache still invalidates correctly when filters change.
    frame = _frame
    G = nx.Graph()
    for _, row in frame.iterrows():
        G.add_node(
            row["Company_Name"],
            type="company",
            market_cap=row["Market_Cap_B"],
            sector=row["Sector"],
            emb_x=row["Emb_X"],
            emb_y=row["Emb_Y"],
            segments=row["Segment_List"],
            involvements=row["Involvements"],
            core_products=row["Core_Products"],
        )

    if include_segments:
        seg_nodes = set()
        for segs in frame["Segment_List"]:
            seg_nodes.update(segs)
        for seg in seg_nodes:
            seg_companies = frame[frame["Segment_List"].apply(lambda x: seg in x)]
            avg_x = seg_companies["Emb_X"].mean()
            avg_y = seg_companies["Emb_Y"].mean()
            G.add_node(seg, type="segment", emb_x=avg_x, emb_y=avg_y, coverage=len(seg_companies))

    rows = frame.reset_index(drop=True)
    sim_edges = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1, r2 = rows.iloc[i], rows.iloc[j]
            dist = np.sqrt((r1["Emb_X"] - r2["Emb_X"]) ** 2 + (r1["Emb_Y"] - r2["Emb_Y"]) ** 2)
            if dist < threshold:
                shared_segs = sorted(set(r1["Segment_List"]) & set(r2["Segment_List"]))
                shared_core = sorted(set(r1["Core_Products"]) & set(r2["Core_Products"]))
                G.add_edge(
                    r1["Company_Name"], r2["Company_Name"],
                    kind="similarity", distance=round(dist, 2),
                    shared_segments=shared_segs, shared_core_products=shared_core,
                )
                sim_edges.append((r1["Company_Name"], r2["Company_Name"]))

    if include_segments:
        for _, row in rows.iterrows():
            for seg in row["Segment_List"]:
                involvement = row["Involvements"].get(seg, None)
                G.add_edge(row["Company_Name"], seg, kind="membership", involvement=involvement)

    return G, len(sim_edges)

with st.spinner("Building network..."):
    _cache_key = tuple(zip(df_year["Company_Name"], df_year["Year"]))
    G, n_sim_edges = build_graph(df_year, _cache_key, edge_threshold, show_segment_nodes)

# Optional spring layout instead of raw embedding coordinates
if layout_mode == "Force-directed (spring)":
    spring_pos = nx.spring_layout(G, seed=42, k=0.6)
    for n, (x, y) in spring_pos.items():
        G.nodes[n]["emb_x"], G.nodes[n]["emb_y"] = x, y

comp_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "company"]
seg_nodes_all = [n for n in G.nodes() if G.nodes[n]["type"] == "segment"]

# ---------------------------------------------------------------------------
# Focus node state -- driven by THREE inputs that all write to the same place:
# the sidebar dropdown, clicking a node in the chart, and the "highlight" button
# in the company-details panel. Whichever fires last wins.
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎯 Focus")

focus_options = ["(None)"] + sorted(comp_nodes)
if "_pending_focus" in st.session_state:
    pending = st.session_state.pop("_pending_focus")
    if pending in focus_options:
        st.session_state["focus_select"] = pending

focus_choice = st.sidebar.selectbox(
    "Focus a company (or click a node below)",
    focus_options,
    key="focus_select",
)
highlighted = None if focus_choice == "(None)" else focus_choice
neighbors = set(G.neighbors(highlighted)) if highlighted else set()

st.sidebar.caption("💡 Tip: click any node in the chart to focus it directly.")

# ---------------------------------------------------------------------------
# Build Plotly figure
# ---------------------------------------------------------------------------
fig = go.Figure()
dim_opacity = 0.12

def edge_coords(edge_list):
    ex, ey = [], []
    for u, v in edge_list:
        x0, y0 = G.nodes[u]["emb_x"], G.nodes[u]["emb_y"]
        x1, y1 = G.nodes[v]["emb_x"], G.nodes[v]["emb_y"]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
    return ex, ey

sim_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "similarity"]
mem_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "membership"]

if highlighted:
    hi_edges = [(u, v) for u, v in G.edges() if highlighted in (u, v)]
    bg_sim = [e for e in sim_edges if e not in hi_edges and tuple(reversed(e)) not in hi_edges]
    bg_mem = [e for e in mem_edges if e not in hi_edges and tuple(reversed(e)) not in hi_edges]
else:
    hi_edges, bg_sim, bg_mem = [], sim_edges, mem_edges

# bucket background similarity edges by strength (closer = stronger)
if bg_sim:
    dists = [G.edges[e]["distance"] for e in bg_sim]
    q1, q2 = np.percentile(dists, [33, 66])
    tiers = [
        ("Strong similarity", lambda d: d <= q1, dict(width=2.0, color="#6c757d")),
        ("Moderate similarity", lambda d: q1 < d <= q2, dict(width=1.1, color="#adb5bd")),
        ("Weak similarity", lambda d: d > q2, dict(width=0.6, color="#dee2e6")),
    ]
    for label, cond, style in tiers:
        tier_edges = [e for e in bg_sim if cond(G.edges[e]["distance"])]
        if not tier_edges:
            continue
        ex, ey = edge_coords(tier_edges)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines", line=style,
            opacity=0.5 if highlighted else 1.0,
            hoverinfo="skip", showlegend=(not highlighted), name=label,
        ))

if show_segment_nodes:
    ex, ey = edge_coords(bg_mem)
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.6, color="#dddddd", dash="dot"),
        opacity=0.4 if highlighted else 0.8,
        hoverinfo="skip", showlegend=False, name="Segment membership"
    ))

if highlighted:
    ex, ey = edge_coords(hi_edges)
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=2.2, color="#e63946"),
        hoverinfo="skip", showlegend=False, name="Connections"
    ))

# invisible midpoint markers carry hover info per edge
def edge_hover_trace(edge_list, text_fn, marker_size=10):
    mx, my, mtext = [], [], []
    for u, v in edge_list:
        x0, y0 = G.nodes[u]["emb_x"], G.nodes[u]["emb_y"]
        x1, y1 = G.nodes[v]["emb_x"], G.nodes[v]["emb_y"]
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        mtext.append(text_fn(u, v))
    return go.Scatter(
        x=mx, y=my, mode="markers",
        marker=dict(size=marker_size, color="rgba(0,0,0,0)"),
        hovertext=mtext, hoverinfo="text", showlegend=False,
    )

def sim_hover_text(u, v):
    d = G.edges[u, v]["distance"]
    shared = G.edges[u, v]["shared_segments"]
    core = G.edges[u, v]["shared_core_products"]
    shared_str = ", ".join(shared) if shared else "none in common"
    core_str = ", ".join(core) if core else "none in common"
    return (f"{u} ↔ {v}<br>Embedding distance: {d:.2f}"
            f"<br>Shared segments: {shared_str}<br>Shared core products: {core_str}")

def mem_hover_text(u, v):
    company, seg = (u, v) if G.nodes[u]["type"] == "company" else (v, u)
    w = G.edges[u, v]["involvement"]
    w_str = f"{w:.2f}" if w is not None else "n/a"
    return f"{company} → {seg}<br>Involvement score: {w_str}"

hover_sim_edges = hi_edges if highlighted else sim_edges
hover_sim_edges = [e for e in hover_sim_edges if G.nodes[e[0]]["type"] == "company" and G.nodes[e[1]]["type"] == "company"]
if hover_sim_edges:
    fig.add_trace(edge_hover_trace(hover_sim_edges, sim_hover_text))

if show_segment_nodes:
    hover_mem_edges = hi_edges if highlighted else mem_edges
    hover_mem_edges = [e for e in hover_mem_edges if "membership" == G.edges[e[0], e[1]].get("kind")]
    if hover_mem_edges:
        fig.add_trace(edge_hover_trace(hover_mem_edges, mem_hover_text, marker_size=8))

# company nodes -- one trace, customdata carries the name so clicks can be resolved
comp_x = [G.nodes[n]["emb_x"] for n in comp_nodes]
comp_y = [G.nodes[n]["emb_y"] for n in comp_nodes]
comp_color = [sector_colors[G.nodes[n]["sector"]] for n in comp_nodes]
comp_size = [max(18, (G.nodes[n]["market_cap"] / 40) + 18) for n in comp_nodes]
comp_text = [
    f"{n}<br>MCap: ${G.nodes[n]['market_cap']:.1f}B<br>Sector: {G.nodes[n]['sector']}"
    f"<br>Segments: {', '.join(G.nodes[n]['segments'])}"
    f"<br>Core products: {', '.join(G.nodes[n]['core_products'])}"
    f"<br><i>Click to focus</i>"
    for n in comp_nodes
]

if highlighted:
    comp_opacity = [1.0 if n == highlighted else (0.95 if n in neighbors else dim_opacity) for n in comp_nodes]
    comp_line_width = [3 if n == highlighted else (1.5 if n in neighbors else 0.5) for n in comp_nodes]
    comp_line_color = ["#e63946" if n == highlighted else "white" for n in comp_nodes]
else:
    comp_opacity = [0.9] * len(comp_nodes)
    comp_line_width = [1.2] * len(comp_nodes)
    comp_line_color = ["white"] * len(comp_nodes)

fig.add_trace(go.Scatter(
    x=comp_x, y=comp_y, mode="markers+text" if show_labels else "markers",
    marker=dict(
        size=comp_size, color=comp_color, opacity=comp_opacity,
        line=dict(width=comp_line_width, color=comp_line_color),
    ),
    text=[n for n in comp_nodes] if show_labels else None,
    textposition="top center",
    hovertext=comp_text, hoverinfo="text",
    customdata=[[n] for n in comp_nodes],
    showlegend=False, name="Companies",
))

# dummy legend traces, one per sector present in view
present_sectors = sorted(df_year["Sector"].unique())
for sec in present_sectors:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=sector_colors[sec]),
        name=sec, showlegend=True
    ))

# segment nodes -- also clickable, also carry customdata
if show_segment_nodes:
    seg_x = [G.nodes[n]["emb_x"] for n in seg_nodes_all]
    seg_y = [G.nodes[n]["emb_y"] for n in seg_nodes_all]
    if highlighted:
        seg_opacity = [1.0 if n in neighbors else dim_opacity for n in seg_nodes_all]
    else:
        seg_opacity = [0.9] * len(seg_nodes_all)
    seg_hover = [f"{n}<br>Companies covering this segment: {G.nodes[n]['coverage']}<br><i>Click to focus</i>" for n in seg_nodes_all]
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y, mode="markers+text",
        marker=dict(size=16, color="#333333", symbol="diamond",
                     opacity=seg_opacity, line=dict(width=1.5, color="white")),
        text=seg_nodes_all, textposition="bottom center",
        hovertext=seg_hover, hoverinfo="text",
        customdata=[[n] for n in seg_nodes_all],
        showlegend=True, name="Segments (diamond)"
    ))

fig.update_layout(
    title=f"{year} Company Network ({len(comp_nodes)} companies, {n_sim_edges} similarity edges)"
          + (f" — focused on {highlighted}" if highlighted else ""),
    showlegend=True,
    legend=dict(title="Sector", orientation="v", x=1.02, y=1),
    hovermode="closest",
    clickmode="event+select",
    xaxis=dict(showgrid=True, zeroline=True, showticklabels=False),
    yaxis=dict(showgrid=True, zeroline=True, showticklabels=False),
    plot_bgcolor="white",
    height=700,
    margin=dict(r=180),
)

click_event = st.plotly_chart(
    fig, width='stretch',
    on_select="rerun", selection_mode="points", key="network_chart",
)

# resolve click -> focus. Runs after render, takes effect on next rerun.
sel_points = (click_event or {}).get("selection", {}).get("points", [])
if sel_points:
    cd = sel_points[0].get("customdata")
    clicked_name = cd[0] if isinstance(cd, list) and cd else cd
    if clicked_name and clicked_name != highlighted:
        st.session_state["_pending_focus"] = clicked_name
        st.rerun()

if highlighted:
    comp_neighbors = sorted(n for n in neighbors if G.nodes[n]["type"] == "company")
    seg_neighbors = sorted(n for n in neighbors if G.nodes[n]["type"] == "segment")
    with st.expander(f"🔗 Connections for {highlighted} ({len(neighbors)} total)", expanded=True):
        c1, c2 = st.columns(2)
        c1.write("**Connected companies**")
        c1.write(", ".join(comp_neighbors) if comp_neighbors else "None")
        c2.write("**Shared segments**")
        c2.write(", ".join(seg_neighbors) if seg_neighbors else "None")

# ---------------------------------------------------------------------------
# Network Story -- plain-language explanation of how a company sits in the
# network, before we let anyone start pulling pieces out of it.
# ---------------------------------------------------------------------------
st.header("🧭 Network Story")

story_default = highlighted if highlighted in comp_nodes else df_year.iloc[0]["Company_Name"]
story_company = st.selectbox(
    "Tell the story for", sorted(comp_nodes),
    index=sorted(comp_nodes).index(story_default), key="story_company_select",
)

def segment_coverage(seg):
    return G.nodes[seg]["coverage"] if seg in G.nodes and G.nodes[seg]["type"] == "segment" else \
        int(df_year["Segment_List"].apply(lambda x: seg in x).sum())

node = G.nodes[story_company]
seg_list = node["segments"]
core_list = node["core_products"]

# rank this company's own segments by how rare/crowded they are in the current view
seg_coverage_pairs = sorted(((seg, segment_coverage(seg)) for seg in seg_list), key=lambda t: t[1])
rarest = seg_coverage_pairs[0] if seg_coverage_pairs else None
most_crowded = seg_coverage_pairs[-1] if seg_coverage_pairs else None

# rank neighbors by embedding distance
story_neighbors = []
for u, v, d in G.edges(data=True):
    if d.get("kind") != "similarity":
        continue
    if story_company not in (u, v):
        continue
    other = v if u == story_company else u
    story_neighbors.append((other, d["distance"], d["shared_segments"], d["shared_core_products"]))
story_neighbors.sort(key=lambda t: t[1])
top_neighbors = story_neighbors[:5]

total_mcap_view = df_year["Market_Cap_B"].sum()
# illustrative-only dependency score: how much of this company's closest-neighbor
# footprint is concentrated in companies that also carry meaningful market cap weight.
# NOT a financial estimate -- purely a relative, in-dataset heuristic for the prototype.
dep_score = 0.0
if top_neighbors and node["segments"]:
    for other, dist, shared, _ in top_neighbors:
        overlap_ratio = len(shared) / len(node["segments"])
        mcap_weight = df_year.loc[df_year["Company_Name"] == other, "Market_Cap_B"].sum() / total_mcap_view
        dep_score += overlap_ratio * mcap_weight
    dep_score = min(100.0, dep_score * 100)

st.markdown(f"**{story_company}** operates across **{len(seg_list)} segments** "
            f"({', '.join(seg_list)}), with **{len(core_list)} core product lines** highlighted in its filing.")

if top_neighbors:
    lines = []
    for other, dist, shared, core in top_neighbors:
        shared_str = ", ".join(shared) if shared else "no declared segments"
        lines.append(f"- **{other}** — embedding distance {dist:.2f}, shares *{shared_str}*")
    st.markdown("Its closest peers by business-description similarity are:\n" + "\n".join(lines))
else:
    st.markdown("No other company falls within the current similarity threshold — it sits alone in the embedding space at this setting.")

if rarest and most_crowded:
    rarity_note = (
        f"Within this filtered view, **{story_company}**'s presence in **{rarest[0]}** is comparatively rare "
        f"(shared with only {rarest[1]} other compan{'y' if rarest[1]==1 else 'ies'}), while its position in "
        f"**{most_crowded[0]}** is crowded (shared with {most_crowded[1]} companies)."
    )
    st.markdown(rarity_note)

st.markdown(
    f"🧪 **Illustrative Network Dependency Score: {dep_score:.0f}/100** — a heuristic combining segment overlap "
    "and neighbor market-cap weight for this prototype. *Not a real financial estimate.*"
)

st.caption(
    "This story is meant to set up the what-if analysis below: it names which companies and segments "
    f"**{story_company}**'s network position currently leans on most."
)

# ---------------------------------------------------------------------------
# Company details section
# ---------------------------------------------------------------------------
st.header("📊 Company Details")
default_idx = sorted(comp_nodes).index(highlighted) if highlighted else 0
selected_company = st.selectbox("Select a Company", sorted(comp_nodes), index=default_idx, key="details_select")

if selected_company:
    company_data = df_year[df_year["Company_Name"] == selected_company].iloc[0]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Info")
        st.write(f"**GVKEY**: {company_data['GVKEY']}")
        st.write(f"**CIK**: {company_data['CIK']}")
        st.write(f"**Sector**: {company_data['Sector']}")
        st.write(f"**Market Cap**: ${company_data['Market_Cap_B']:.2f}B")
        st.write(f"**Segments**: {', '.join(company_data['Segment_List'])}")
        st.write(f"**Core products**: {', '.join(company_data['Core_Products'])}")

    with col2:
        st.subheader("Embedding Position")
        st.write(f"**X Coordinate**: {company_data['Emb_X']:.4f}")
        st.write(f"**Y Coordinate**: {company_data['Emb_Y']:.4f}")
        st.write("**Item 1 Snippet**:")
        st.info(company_data["Item_1_Snippet"])

    if st.button(f"🎯 Focus {selected_company} in the network above"):
        st.session_state["_pending_focus"] = selected_company
        st.rerun()

# ---------------------------------------------------------------------------
# What-if analysis
# ---------------------------------------------------------------------------
st.header("🧪 What-If Analysis")
st.caption(
    "Prototype-only simulation. It approximates impact using the segment / core-product overlap "
    "already in the data — it does not recompute real embeddings, and none of the scores below are "
    "real financial estimates. Treat them as directional, illustrative signals only."
)

whatif_mode = st.radio(
    "Simulate removing:", ["A company from the network", "A product/service line from a company"],
    horizontal=True,
)

if whatif_mode == "A company from the network":
    remove_company = st.selectbox(
        "Company to remove", sorted(comp_nodes),
        index=sorted(comp_nodes).index(story_company), key="remove_company_select",
    )
    rc_node = G.nodes[remove_company]
    rc_segments = set(rc_node["segments"])
    rc_mcap = rc_node["market_cap"]

    affected = []
    for u, v, d in G.edges(data=True):
        if d.get("kind") != "similarity" or remove_company not in (u, v):
            continue
        other = v if u == remove_company else u
        other_segs = set(G.nodes[other]["segments"])
        shared = sorted(rc_segments & other_segs)
        overlap_ratio = len(shared) / len(other_segs) if other_segs else 0
        exposure = round(overlap_ratio * (rc_mcap / total_mcap_view) * 100, 2)
        affected.append({
            "Company": other,
            "Shared segments lost": ", ".join(shared) if shared else "(similarity only, no shared segment)",
            "Similarity edges lost": 1,
            "Illustrative exposure %": exposure,
        })

    affected_cols = ["Company", "Shared segments lost", "Similarity edges lost", "Illustrative exposure %"]
    affected_df = pd.DataFrame(affected, columns=affected_cols)
    if not affected_df.empty:
        affected_df = affected_df.sort_values("Illustrative exposure %", ascending=False)

    seg_impact = []
    for seg in sorted(rc_segments):
        before = segment_coverage(seg)
        after = before - 1
        seg_impact.append({"Segment": seg, "Coverage before": before, "Coverage after": after,
                            "Orphaned?": "⚠️ Yes" if after == 0 else "No"})
    seg_impact_df = pd.DataFrame(seg_impact)

    m1, m2, m3 = st.columns(3)
    m1.metric("Companies directly connected", len(affected_df))
    m2.metric("Similarity edges removed", len(affected_df))
    m3.metric("Segments this company touched", len(rc_segments))

    if not affected_df.empty:
        st.markdown(f"**Most exposed companies if {remove_company} exits:**")
        st.dataframe(affected_df.head(10), width='stretch', hide_index=True)
        top = affected_df.iloc[0]
        st.markdown(
            f"⚠️ Illustratively, **{top['Company']}** looks most exposed, with an exposure score of "
            f"**{top['Illustrative exposure %']:.1f}**, driven by overlap in *{top['Shared segments lost']}*."
        )
    else:
        st.markdown(f"No other company in the current view is similarity-connected to **{remove_company}**.")

    st.markdown("**Segment coverage impact:**")
    st.dataframe(seg_impact_df, width='stretch', hide_index=True)
    orphaned = seg_impact_df[seg_impact_df["Orphaned?"] != "No"]
    if not orphaned.empty:
        st.markdown(f"⚠️ Removing **{remove_company}** would leave **{', '.join(orphaned['Segment'])}** "
                    "with zero remaining coverage in this filtered view.")

else:
    pick_company = st.selectbox(
        "Company", sorted(comp_nodes),
        index=sorted(comp_nodes).index(story_company), key="remove_product_company_select",
    )
    pc_core = G.nodes[pick_company]["core_products"]
    if not pc_core:
        st.info(f"{pick_company} has no listed core product lines to remove.")
    else:
        pick_product = st.selectbox("Product / service line to remove", pc_core, key="remove_product_select")

        affected = []
        for other in comp_nodes:
            if other == pick_company:
                continue
            other_core = set(G.nodes[other]["core_products"])
            if pick_product not in other_core:
                continue
            shared_before = set(pc_core) & other_core
            shared_after = shared_before - {pick_product}
            other_mcap = G.nodes[other]["market_cap"]
            status = "🔴 Fully disconnected (core products)" if not shared_after else f"🟡 Weakened — still shares {', '.join(sorted(shared_after))}"
            exposure = round((1 / len(shared_before)) * (G.nodes[pick_company]["market_cap"] / total_mcap_view) * 100, 2)
            affected.append({
                "Company": other,
                "Status": status,
                "Illustrative exposure %": exposure,
            })

        affected_cols = ["Company", "Status", "Illustrative exposure %"]
        affected_df = pd.DataFrame(affected, columns=affected_cols)
        if not affected_df.empty:
            affected_df = affected_df.sort_values("Illustrative exposure %", ascending=False)
        coverage_before = int(df_year["Core_Products"].apply(lambda x: pick_product in x).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Companies sharing this product line", coverage_before)
        m2.metric("Would fully disconnect", int((affected_df["Status"].str.startswith("🔴")).sum()) if not affected_df.empty else 0)
        m3.metric("Would just weaken", int((affected_df["Status"].str.startswith("🟡")).sum()) if not affected_df.empty else 0)

        if not affected_df.empty:
            st.markdown(f"**Companies affected if {pick_company} drops \"{pick_product}\":**")
            st.dataframe(affected_df, width='stretch', hide_index=True)
            fully_gone = affected_df[affected_df["Status"].str.startswith("🔴")]
            if not fully_gone.empty:
                st.markdown(
                    f"⚠️ **{', '.join(fully_gone['Company'])}** would lose their *only* core-product overlap "
                    f"with **{pick_company}** — illustratively, this is where dropping \"{pick_product}\" "
                    "would be felt most in this network."
                )
        else:
            st.markdown(f"No other company in the current view shares **\"{pick_product}\"** as a core product line "
                        f"with **{pick_company}** — removing it wouldn't change any core-product connections here.")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⬇️ Export")
csv = df_year.drop(columns=["Segment_List", "Core_Products"]).to_csv(index=False)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name=f"10k_network_{year}_filtered.csv",
    mime="text/csv",
)