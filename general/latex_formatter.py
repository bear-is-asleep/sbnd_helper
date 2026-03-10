"""Utilities for rendering systematic-uncertainty data as LaTeX tables."""
import os


def escape_latex(s):
    """Escape special LaTeX characters in a string."""
    if s is None:
        return ''
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def fmt_unc(val):
    """Format a fractional uncertainty as a percentage string."""
    if val is not None and not hasattr(val, '__len__'):
        return f'{val * 100:.2f}'
    return '{-}'


def _safe_desc(desc):
    """Return desc as a string, handling lists or None."""
    if not desc or not isinstance(desc, str):
        return ''
    return desc


def _is_named_tables(template):
    """Check if the input is a dict of named templates (multi-table) vs a single template."""
    if not isinstance(template, dict) or len(template) == 0:
        return False
    first_val = next(iter(template.values()))
    if not isinstance(first_val, dict):
        return False
    inner = next(iter(first_val.values()), None)
    return isinstance(inner, dict) and 'name' in inner


def _resolve_tables(template):
    """Normalize input into (named_tables dict, sorted_keys list, is_multi bool)."""
    multi = _is_named_tables(template)
    named_tables = template if multi else {None: template}
    tbl_names = list(named_tables.keys())

    all_keys = {}
    for tbl in named_tables.values():
        for key, entry in tbl.items():
            if key not in all_keys:
                all_keys[key] = entry.get('order', 0) or 0
    sorted_keys = sorted(all_keys.keys(), key=lambda k: (all_keys[k], k))
    return named_tables, tbl_names, sorted_keys, multi


def _begin_longtblr(col_spec, caption=None, label=None):
    """Return the opening lines of a longtblr environment."""
    options = []
    if caption:
        options.append(f'  caption = {{{caption}}}')
    if label:
        options.append(f'  label = {{{label}}}')
    options_block = ',\n'.join(options)

    lines = [r'{\tiny']
    if options_block:
        lines.append(r'\begin{longtblr}[')
        lines.append(options_block)
        lines.append(rf']{{colspec={{{col_spec}}}}}')
    else:
        lines.append(rf'\begin{{longtblr}}{{colspec={{{col_spec}}}}}')
    lines.append(r'\hline')
    return lines


def _save(result, save_dir, filename):
    """Write result to file if save_dir is set."""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        with open(path, 'w') as f:
            f.write(result)
        print(f'LaTeX table saved to {path}')


def description_table(template, caption=None, label=None,
                      do_escape=True, save_dir=None,
                      filename='systematics_descriptions.tex'):
    """
    Build a LaTeX longtblr with Name, Type, Description columns only.

    Parameters
    ----------
    template : dict
        Single template dict from ``generate_description_template``.
    caption : str, optional
    label : str, optional
    do_escape : bool
    save_dir : str, optional
    filename : str

    Returns
    -------
    str
    """
    _esc = escape_latex if do_escape else (lambda s: s or '')

    sorted_items = sorted(
        template.items(),
        key=lambda x: (x[1].get('order', 0) or 0, x[0])
    )

    col_spec = 'lll'
    lines = _begin_longtblr(col_spec, caption=caption, label=label)
    lines.append(r'Name & Type & Description \\')
    lines.append(r'\hline')

    for key, entry in sorted_items:
        name = _esc(entry.get('name', key))
        variation = _esc(entry.get('variation', ''))
        description = _safe_desc(entry.get('description', ''))
        lines.append(rf'{name} & {variation} & {description} \\')

    lines.append(r'\hline')
    lines.append(r'\end{longtblr}}')
    result = '\n'.join(lines)
    _save(result, save_dir, filename)
    return result


def uncertainty_table(template, include_unc='both', caption=None, label=None,
                      do_escape=True, save_dir=None,
                      filename='systematics_uncertainties.tex'):
    """
    Build a LaTeX longtblr with Name and uncertainty columns.

    Accepts a single template or a dict of named templates for horizontal
    merging (e.g. ``{r'$\\cos\\theta_\\mu$': t1, r'$p_\\mu$': t2}``).

    Parameters
    ----------
    template : dict
        Single template or ``{column_label: template_dict}``.
    include_unc : str
        'event', 'xsec', or 'both'.
    caption : str, optional
    label : str, optional
    do_escape : bool
    save_dir : str, optional
    filename : str

    Returns
    -------
    str
    """
    _esc = escape_latex if do_escape else (lambda s: s or '')
    named_tables, tbl_names, sorted_keys, multi = _resolve_tables(template)

    unc_fields = []
    if include_unc in ('event', 'both'):
        unc_fields.append('event_rate_unc')
    if include_unc in ('xsec', 'both'):
        unc_fields.append('xsec_unc')

    unc_labels = {
        'event_rate_unc': r'Event (\%)',
        'xsec_unc': r'Xsec (\%)',
    }

    unc_columns = []
    for tname in tbl_names:
        for field in unc_fields:
            if multi:
                col_header = rf'\shortstack{{{tname} \\ {unc_labels[field]}}}'
            else:
                col_header = unc_labels[field]
            unc_columns.append((tname, field, col_header))

    col_spec = 'l' + 'c' * len(unc_columns)
    lines = _begin_longtblr(col_spec, caption=caption, label=label)

    header = 'Name'
    for _, _, col_header in unc_columns:
        header += f' &\n{col_header}'
    header += r' \\'
    lines.append(header)
    lines.append(r'\hline')

    for key in sorted_keys:
        ref_entry = None
        for tbl in named_tables.values():
            if key in tbl:
                ref_entry = tbl[key]
                break

        name = _esc(ref_entry.get('name', key))
        row = name
        for tname, field, _ in unc_columns:
            entry = named_tables[tname].get(key, {})
            row += f' & {fmt_unc(entry.get(field, None))}'
        row += r' \\'
        lines.append(row)

    lines.append(r'\hline')
    lines.append(r'\end{longtblr}}')
    result = '\n'.join(lines)
    _save(result, save_dir, filename)
    return result