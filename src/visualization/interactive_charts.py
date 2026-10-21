import plotly.graph_objects as go


def running_best(history):
    values = []
    best = float('-inf')
    for value in history:
        best = max(best, value)
        values.append(best)
    return values


def _style(fig, x_title, y_title):
    fig.update_layout(
        template='plotly_white',
        height=365,
        margin=dict(l=20, r=20, t=22, b=40),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(100,116,139,0.14)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(100,116,139,0.14)')
    return fig


def _line(fig, values, name, *, dash='solid', width=2.5):
    fig.add_trace(go.Scatter(
        x=list(range(len(values))), y=list(values), name=name,
        mode='lines', line=dict(dash=dash, width=width),
        hovertemplate=f'{name}<br>Step %{{x}}<br>Score %{{y:,.2f}}<extra></extra>',
    ))


def optimization_figure(result):
    fig = go.Figure()
    is_ga = result.algorithm == 'Genetic Algorithm'
    if is_ga:
        _line(fig, result.history, 'Generation Best')
        avg = result.metadata.get('average_fitness_history', [])
        if avg:
            _line(fig, avg, 'Generation Average')
        _line(fig, running_best(result.history), 'Best So Far', dash='dash')
    else:
        _line(fig, result.history, 'Current Score')
        _line(fig, running_best(result.history), 'Best So Far', dash='dash')
    return _style(fig, 'Generation' if is_ga else 'Iteration', 'Objective Score')


def sa_diagnostic_figure(values, title, y_title, *, probability=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(values) + 1)), y=list(values),
        mode='lines', name=title,
        hovertemplate=f'Iteration %{{x}}<br>{y_title}: %{{y:.4f}}<extra></extra>',
    ))
    _style(fig, 'Iteration', y_title)
    fig.update_layout(height=285, showlegend=False, title=dict(text=title, font=dict(size=14)))
    if probability:
        fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def comparison_figure(results):
    fig = go.Figure()
    # Marks and styles make coincident HC curves easier to distinguish.
    styles = {
        'Hill Climbing': ('solid', 'circle'),
        'Hill Climbing (Sideways)': ('dash', 'square'),
        'Hill Climbing (Stochastic)': ('dot', 'diamond'),
        'Hill Climbing (Random Restart)': ('dashdot', 'triangle-up'),
        'Simulated Annealing': ('solid', 'cross'),
        'Genetic Algorithm': ('longdash', 'x'),
    }
    for name, result in results.items():
        values = running_best(result.history)
        dash, symbol = styles.get(name, ('solid', 'circle'))
        # sparse markers to expose duplicate lines without overwhelming chart
        stride = max(1, len(values) // 14)
        marker_sizes = [7 if i % stride == 0 or i == len(values)-1 else 0
                        for i in range(len(values))]
        fig.add_trace(go.Scatter(
            x=list(range(len(values))), y=values,
            mode='lines+markers', name=name,
            line=dict(dash=dash, width=2.5),
            marker=dict(symbol=symbol, size=marker_sizes),
            hovertemplate=f'{name}<br>Step %{{x}}<br>Best Objective %{{y:,.2f}}<extra></extra>',
        ))
    return _style(fig, 'Iteration / Generation', 'Best Objective Score')
