from __future__ import annotations

import math
import plotly.graph_objects as go


def _rounded_outline(x0, y0, x1, y1, radius, steps=5):
    """Rounded closed polygon coordinates (in data/grid units)."""
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    corners = [
        (x1-r, y0+r, -math.pi/2, 0),
        (x1-r, y1-r, 0, math.pi/2),
        (x0+r, y1-r, math.pi/2, math.pi),
        (x0+r, y0+r, math.pi, 3*math.pi/2),
    ]
    xs, ys = [], []
    for cx, cy, start, end in corners:
        for i in range(steps+1):
            a = start + (end-start)*i/steps
            xs.append(cx + r*math.cos(a))
            ys.append(cy + r*math.sin(a))
    xs.append(xs[0]); ys.append(ys[0])
    return xs, ys


def ship_figure(state, vehicles, ship, title="Ship Layout"):
    """Return Plotly figure; marker-trace customdata contains a vehicle ID."""
    fig = go.Figure()
    palette = [
        '#60A5FA', '#34D399', '#FBBF24', '#F472B6', '#A78BFA',
        '#22D3EE', '#FB923C', '#4ADE80', '#93C5FD', '#C4B5FD',
    ]

    for index, vid in enumerate(sorted(vehicles)):
        placement = state.placements.get(vid)
        if placement is None:
            continue
        vehicle = vehicles[vid]
        width, length = placement.dimension(vehicle)
        # Geometrical placement unchanged. Insets only affect visual rendering.
        inset = 0.10
        x0, y0 = placement.x + inset, placement.y + inset
        x1, y1 = placement.x + width - inset, placement.y + length - inset
        xs, ys = _rounded_outline(x0, y0, x1, y1, radius=0.22)
        orientation = getattr(placement.orientation, 'value', str(placement.orientation))
        detail = (
            f'<b>{vid}</b><br>'
            f'Position: ({placement.x}, {placement.y})<br>'
            f'Orientation: {orientation}<br>'
            f'Footprint: {width} × {length} cells<br>'
            f'Vehicle: {vehicle.width} × {vehicle.length}<br>'
            f'Weight: {vehicle.weight:g}<br>'
            f'Shipping Fee: {vehicle.shipping_fee:,.0f}<br>'
            f'ETA: {vehicle.eta}'
        )
        color = palette[index % len(palette)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            line=dict(color='rgba(20,35,54,0.55)', width=1.3),
            fillcolor=color, name=vid, showlegend=False,
            hoveron='fills', hovertemplate=detail + '<extra></extra>',
        ))
        # Center hit target for Streamlit selection/click; polygon itself is hoverable.
        cx, cy = placement.x + width/2, placement.y + length/2
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text', text=[vid], textposition='middle center',
            textfont=dict(size=10, color='#172334'),
            marker=dict(size=22, color='rgba(255,255,255,0.01)'),
            customdata=[[vid]],
            name=vid, showlegend=False,
            hovertemplate=detail+'<extra></extra>',
            selected=dict(marker=dict(opacity=0.15)),
            unselected=dict(marker=dict(opacity=0.01)),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        margin=dict(l=55, r=20, t=45, b=55),
        height=max(480, min(800, int(ship.length * 34))),
        template='plotly_white', showlegend=False,
        dragmode='pan', hovermode='closest',
        xaxis=dict(
            range=[0, ship.width], autorange=False,
            tickmode='linear', tick0=0, dtick=1,
            title='Width (x)', showgrid=True, gridcolor='#E2E8F0',
            zeroline=False, fixedrange=False,
        ),
        yaxis=dict(
            range=[ship.length, 0], autorange=False,
            tickmode='linear', tick0=0, dtick=1,
            title='Length (y)', showgrid=True, gridcolor='#E2E8F0',
            zeroline=False, fixedrange=False,
            scaleanchor='x', scaleratio=1,
        ),
    )
    return fig


def selected_vehicle_id(event):
    """Extract a selected ID from Streamlit PlotlyState, if selection occurred."""
    if event is None:
        return None
    try:
        points = event.selection.points
    except (AttributeError, KeyError, TypeError):
        return None
    for point in points:
        payload = point.get('customdata') if hasattr(point, 'get') else None
        if isinstance(payload, (list, tuple)) and payload:
            return str(payload[0])
    return None
