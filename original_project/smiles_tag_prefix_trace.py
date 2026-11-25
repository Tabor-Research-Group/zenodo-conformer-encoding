"""Inspect cumulative block prefixes of one existing compressed SMILES tag.

This module never selects a submolecule or rebuilds a conformer. Pass an
already-generated canonical tagged SMILES, for example ``make_smi(smi)``.
With the default compressed 16-bit format, the first atom contributes no
bytes, the second contributes two, and each subsequent atom contributes four.
The intermediate strings are byte prefixes of that tag, not independently
generated tags for the corresponding molecular fragments.
"""

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from html import escape

import numpy as np
from rdkit import Chem
from matplotlib.colors import to_hex, to_rgb
from McUtils.Graphs import EdgeGraph
from McUtils.ExternalPrograms import RDMolecule


@dataclass(frozen=True)
class TagPrefixStep:
    number: int
    block: tuple[int, ...]
    cumulative_atoms: int
    byte_start: int
    byte_end: int
    new_bytes: bytes
    block_base64: str
    prefix_tag: str
    unchanged_chars: int
    replaced_old_suffix: str
    added_new_suffix: str
    base_smiles: str = ''


def _byte_count(atom_count, word_bytes):
    """Packed 16-bit tag: 0, 1, 3, then 2 more words per atom."""
    return max(0, 2 * atom_count - 3) * word_bytes


def trace_tag_prefixes(tagged_smi, *, blocks=None):
    """Return one re-encoded base64 byte prefix for each canonical block.

    ``blocks`` defaults to the McUtils canonical fragments of the heavy-atom
    graph parsed from the SMILES before the underscore. If supplied, it must
    partition atom indices in the original tag order, without reordering them.
    """
    core, separator, tag = tagged_smi.partition('_')
    if not separator:
        raise ValueError('Expected a SMILES followed by an underscore and base64 tag')
    mol = Chem.MolFromSmiles(core)
    if mol is None:
        raise ValueError('The chemical SMILES before the tag could not be parsed')
    atom_count = mol.GetNumAtoms()
    if atom_count < 3:
        raise ValueError('The compressed tag layout requires at least three atoms')

    data = base64.b64decode(tag, validate=True)
    if base64.b64encode(data).decode('ascii') != tag:
        raise ValueError('The base64 tag is not in canonical form')
    bit_width = RDMolecule.default_tag_byte_size
    if bit_width != 16:
        raise NotImplementedError('This trace supports the default 16-bit compressed tag')
    word_bytes = bit_width // 8
    expected = _byte_count(atom_count, word_bytes)
    if len(data) != expected:
        raise ValueError(
            f'{atom_count} atoms require {expected} compressed tag bytes; '
            f'found {len(data)}. Check the encoder and atom order.'
        )

    if blocks is None:
        graph = EdgeGraph(
            np.arange(atom_count),
            [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
             for bond in mol.GetBonds()],
        )
        blocks = [fragment for _, fragment in
                  graph.get_canonical_fragments(np.arange(atom_count))]
    blocks = [tuple(int(i) for i in block) for block in blocks]
    flattened = tuple(i for block in blocks for i in block)
    if flattened != tuple(range(atom_count)):
        raise ValueError('Blocks must cover atoms 0..N-1 in tag order')

    steps = []
    previous_end = 0
    previous_tag = ''
    cumulative = 0
    for number, block in enumerate(blocks, start=1):
        cumulative += len(block)
        end = _byte_count(cumulative, word_bytes)
        new_bytes = data[previous_end:end]
        prefix = base64.b64encode(data[:end]).decode('ascii')
        common = 0
        for old_char, new_char in zip(previous_tag, prefix):
            if old_char != new_char:
                break
            common += 1
        steps.append(TagPrefixStep(
            number=number,
            block=block,
            cumulative_atoms=cumulative,
            byte_start=previous_end,
            byte_end=end,
            new_bytes=new_bytes,
            block_base64=base64.b64encode(new_bytes).decode('ascii'),
            prefix_tag=prefix,
            unchanged_chars=common,
            replaced_old_suffix=previous_tag[common:],
            added_new_suffix=prefix[common:],
            base_smiles=core,
        ))
        previous_end = end
        previous_tag = prefix

    if previous_end != len(data) or steps[-1].prefix_tag != tag:
        raise RuntimeError('The final block did not recover the original tag')
    return steps


def _fragment_colors(steps, palette, single_atom_palette):
    """Match plot_lipitor_partition's independent fragment color cycles."""
    if isinstance(palette, str):
        palette = (palette,)
    if isinstance(single_atom_palette, str):
        single_atom_palette = (single_atom_palette,)
    palette = tuple(palette)
    single_atom_palette = tuple(single_atom_palette)
    if not palette or not single_atom_palette:
        raise ValueError('Both fragment palettes must contain at least one color')
    colors = []
    palette_index = 0
    single_atom_index = 0
    for step in steps:
        if len(step.block) == 1:
            colors.append(single_atom_palette[
                single_atom_index % len(single_atom_palette)
            ])
            single_atom_index += 1
        else:
            colors.append(palette[palette_index % len(palette)])
            palette_index += 1
    return colors


def _resolve_fragment_palettes(palette, single_atom_palette, single_atom_color):
    if single_atom_palette is not None and single_atom_color is not None:
        raise ValueError('Pass single_atom_palette or single_atom_color, not both')
    if palette is None or (single_atom_palette is None and single_atom_color is None):
        from plot_lipitor_partition import PALETTE, SINGLE_ATOM_PALETTE
        if palette is None:
            palette = PALETTE
        if single_atom_palette is None and single_atom_color is None:
            single_atom_palette = SINGLE_ATOM_PALETTE
    if single_atom_color is not None:
        single_atom_palette = (single_atom_color,)
    if isinstance(palette, str):
        palette = (palette,)
    if isinstance(single_atom_palette, str):
        single_atom_palette = (single_atom_palette,)
    palette = tuple(palette)
    single_atom_palette = tuple(single_atom_palette)
    if not palette or not single_atom_palette:
        raise ValueError('Both fragment palettes must contain at least one color')
    return palette, single_atom_palette


def _tint(color, white_fraction):
    rgb = to_rgb(color)
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(255 * ((1 - white_fraction) * value + white_fraction))
        for value in rgb
    ))


def _character_owners(character_index, byte_owners):
    """Return the fragment and bit count contributing to one base64 glyph."""
    group_start = (character_index // 4) * 3
    within_group = character_index % 4
    contributions = (
        ((0, 6),), ((0, 2), (1, 4)),
        ((1, 4), (2, 2)), ((2, 6),),
    )[within_group]
    owners = []
    for byte_offset, bit_count in contributions:
        byte_index = group_start + byte_offset
        if byte_index < len(byte_owners):
            owner = byte_owners[byte_index]
            if owners and owners[-1][0] == owner:
                owners[-1] = (owner, owners[-1][1] + bit_count)
            else:
                owners.append((owner, bit_count))
    return owners


def _colored_tag(step, steps, colors):
    byte_owners = [0] * step.byte_end
    for fragment in steps[:step.number]:
        byte_owners[fragment.byte_start:fragment.byte_end] = [fragment.number - 1] * len(fragment.new_bytes)

    glyphs = []
    for index, character in enumerate(step.prefix_tag):
        owners = _character_owners(index, byte_owners)
        if character == '=':
            background = '#edf0f3'
            label = 'base64 padding'
        elif len(owners) == 1:
            owner = owners[0][0]
            background = _tint(colors[owner], .47 if owner == step.number - 1 else .73)
            label = f'fragment {owner + 1}'
        elif len(owners) == 2:
            (first, first_bits), (second, second_bits) = owners
            boundary = round(100 * first_bits / (first_bits + second_bits))
            first_color = _tint(colors[first], .47 if first == step.number - 1 else .73)
            second_color = _tint(colors[second], .47 if second == step.number - 1 else .73)
            background = (
                f'linear-gradient(90deg,{first_color} 0% {boundary}%,'
                f'{second_color} {boundary}% 100%)'
            )
            label = f'boundary: fragment {first + 1} / fragment {second + 1}'
        else:
            background = '#edf0f3'
            label = 'base64 padding'

        underline = (
            f'box-shadow:inset 0 -3px {colors[step.number - 1]};'
            if character != '=' and any(owner == step.number - 1 for owner, _ in owners)
            else ''
        )
        background_style = (
            f'background:{background};' if background.startswith('linear-gradient')
            else f'background-color:{background};'
        )
        glyphs.append(
            '<span title="' + escape(label, quote=True) + '" style="'
            + background_style + underline + 'border-radius:2px;">'
            + escape(character) + '</span>'
        )
        if index % 4 == 3:
            glyphs.append('<wbr>')
    return ''.join(glyphs)


def _powerpoint_runs(step, steps, colors):
    """Editable PowerPoint runs for one prefix, colored by contributing block."""
    byte_owners = [0] * step.byte_end
    for fragment in steps[:step.number]:
        byte_owners[fragment.byte_start:fragment.byte_end] = (
            [fragment.number - 1] * len(fragment.new_bytes)
        )
    runs = []
    for index, character in enumerate(step.prefix_tag):
        owners = _character_owners(index, byte_owners)
        # A base64 glyph can contain bits from two fragments. Its text color
        # comes from the larger contribution; ties go to the newer fragment.
        owner = max(owners, key=lambda item: (item[1], item[0]))[0] if owners else None
        color = colors[owner] if owner is not None and character != '=' else '#7b8490'
        new = owner == step.number - 1 and character != '='
        style = (color, new)
        if runs and (runs[-1]['color'], runs[-1]['new']) == style:
            runs[-1]['text'] += character
        else:
            runs.append({'text': character, 'color': color, 'new': new})
    assert ''.join(run['text'] for run in runs) == step.prefix_tag
    return runs


def _smiles_tokens(smiles, atom_count):
    """Lex SMILES without changing its spelling or RDKit's atom order."""
    atom_chars = set('BCNOPSFIbcnops*')
    two_char_atoms = ('Cl', 'Br', 'se', 'as', 'te')
    tokens = []
    atom_index = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            end = smiles.find(']', i + 1)
            if end < 0:
                raise ValueError('Unclosed bracket atom in base SMILES')
            token = smiles[i:end + 1]
        elif any(smiles.startswith(symbol, i) for symbol in two_char_atoms):
            token = next(symbol for symbol in two_char_atoms
                         if smiles.startswith(symbol, i))
        elif smiles[i] in atom_chars:
            token = smiles[i]
        else:
            if smiles.startswith('%(', i):
                end = smiles.find(')', i + 2)
                if end < 0:
                    raise ValueError('Unclosed ring number in base SMILES')
                token = smiles[i:end + 1]
            elif smiles[i] == '%' and smiles[i + 1:i + 3].isdigit():
                token = smiles[i:i + 3]
            else:
                token = smiles[i]
            tokens.append([token, None])
            i += len(token)
            continue
        tokens.append([token, atom_index])
        atom_index += 1
        i += len(token)
    if atom_index != atom_count:
        raise ValueError(
            f'SMILES lexer found {atom_index} atoms but RDKit found {atom_count}; '
            'cannot safely color the base SMILES'
        )
    return tokens


def _smiles_owners(smiles, steps):
    """Assign atoms and punctuation to fragment indices in the original SMILES."""
    atom_count = sum(len(step.block) for step in steps)
    tokens = _smiles_tokens(smiles, atom_count)
    fragments = [None] * atom_count
    for fragment_index, step in enumerate(steps):
        for atom_index in step.block:
            fragments[atom_index] = fragment_index
    if any(fragment is None for fragment in fragments):
        raise ValueError('Fragments do not cover every SMILES atom')

    previous_atom = None
    owners = []
    for position, (token, atom_index) in enumerate(tokens):
        if atom_index is not None:
            owner_atom = atom_index
            previous_atom = atom_index
        elif token == '(' or token in '-=#:/$\\':
            # An opening parenthesis or bond marker introduces the next atom.
            # A ring marker instead belongs to the preceding atom.
            owner_atom = None
            for following, next_atom in tokens[position + 1:]:
                if next_atom is not None:
                    owner_atom = next_atom
                    break
                if following in (')', '(', '.') or following[0].isdigit() or following.startswith('%'):
                    break
            if owner_atom is None:
                owner_atom = previous_atom
        else:
            # Closing parentheses, ring labels, dots, and other ambiguous
            # punctuation inherit the atom immediately to their left.
            owner_atom = previous_atom
        if owner_atom is None:
            raise ValueError(f'Cannot associate initial SMILES punctuation {token!r}')
        owners.append((token, owner_atom, fragments[owner_atom]))
    return owners


def _smiles_powerpoint_runs(smiles, steps, colors, active_number):
    runs = []
    for token, _, fragment in _smiles_owners(smiles, steps):
        color = colors[fragment]
        new = fragment == active_number - 1
        if runs and (runs[-1]['color'], runs[-1]['new']) == (color, new):
            runs[-1]['text'] += token
        else:
            runs.append({'text': token, 'color': color, 'new': new})
    assert ''.join(run['text'] for run in runs) == smiles
    return runs


def _colored_smiles(smiles, steps, colors):
    return ''.join(
        f'<span title="atom {atom_index}; fragment {fragment + 1}" '
        f'style="background-color:{_tint(colors[fragment], .70)};'
        'border-radius:2px;">' + escape(token) + '</span>'
        for token, atom_index, fragment in _smiles_owners(smiles, steps)
    )


def export_tag_prefixes_pptx(
    steps, output, *, step=-1, all_steps=False,
    palette=None, single_atom_palette=None, single_atom_color=None,
    base_smiles=None,
    node=None, node_modules=None,
):
    """Export colored base SMILES and tag as editable PowerPoint text boxes.

    By default only the final prefix is exported. Set ``step`` to a zero-based
    step index, or ``all_steps=True`` to get one slide per prefix. Open the PPTX
    and copy either text box into another presentation. Their text and colors
    remain editable; visual wrapping does not insert characters into either.

    This uses the bundled JavaScript presentation runtime. ``node`` and
    ``node_modules`` may be supplied if it is installed elsewhere.
    ``single_atom_palette`` cycles independently of ``palette``; the old
    ``single_atom_color`` keyword still sets a one-color singleton palette.
    """
    if not steps:
        raise ValueError('steps must not be empty')
    palette, single_atom_palette = _resolve_fragment_palettes(
        palette, single_atom_palette, single_atom_color
    )
    colors = [to_hex(color) for color in
              _fragment_colors(steps, palette, single_atom_palette)]
    if base_smiles is None:
        base_smiles = getattr(steps[0], 'base_smiles', '')
    if not base_smiles:
        raise ValueError('Pass base_smiles= or regenerate steps with trace_tag_prefixes')
    chosen = list(steps) if all_steps else [steps[step]]
    payload = {
        'slides': [
            {
                'title': f'Conformer tag after fragment {part.number}',
                'base_smiles': base_smiles,
                'smiles_runs': _smiles_powerpoint_runs(
                    base_smiles, steps, colors, part.number),
                'tag': part.prefix_tag,
                'runs': _powerpoint_runs(part, steps, colors),
            }
            for part in chosen
        ]
    }

    runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
    node = Path(node) if node is not None else runtime / 'node/bin/node'
    node_modules = (Path(node_modules) if node_modules is not None
                    else runtime / 'node/node_modules')
    if not node.is_file() or not node_modules.is_dir():
        raise RuntimeError(
            'PowerPoint export needs Node.js and @oai/artifact-tool. '
            'Pass node= and node_modules= paths to the bundled runtime.'
        )
    exporter = Path(__file__).with_name('tag_prefix_to_pptx.mjs')
    if not exporter.is_file():
        raise FileNotFoundError(f'Copy {exporter.name} alongside this module')
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tag-prefix-pptx-') as directory:
        work = Path(directory)
        (work / 'node_modules').symlink_to(node_modules, target_is_directory=True)
        shutil.copy2(exporter, work / exporter.name)
        input_file = work / 'tag.json'
        input_file.write_text(json.dumps(payload), encoding='utf-8')
        subprocess.run(
            [str(node), str(work / exporter.name), str(input_file), str(output)],
            check=True, env=os.environ.copy(),
        )
    return output


def show_tag_prefixes(steps, *, palette=None, single_atom_palette=None,
                      single_atom_color=None, base_smiles=None):
    """Show the growing tag, colored with the molecule's fragment palette.

    The newest fragment has a stronger background and colored underline.
    A glyph receiving bits from bytes in two fragments has a split background.
    ``palette`` can override the plotting palette for a notebook-local style.
    ``single_atom_palette`` cycles for singleton fragments independently;
    ``single_atom_color`` is retained as a one-color alias.
    """
    from IPython.display import HTML, display

    if not steps:
        return
    palette, single_atom_palette = _resolve_fragment_palettes(
        palette, single_atom_palette, single_atom_color
    )
    colors = _fragment_colors(steps, palette, single_atom_palette)
    if base_smiles is None:
        base_smiles = getattr(steps[0], 'base_smiles', '')

    cards = []
    for step, color in zip(steps, colors):
        atoms = (
            f'atom {step.block[0]}' if len(step.block) == 1 else
            f'atoms {step.block[0]}–{step.block[-1]}'
        )
        note = (
            f' · {len(step.replaced_old_suffix)} base64 boundary '
            'character(s) recoded'
            if step.replaced_old_suffix else ''
        )
        tag = _colored_tag(step, steps, colors) or '<em style="color:#697386">No tag bytes yet</em>'
        cards.append(
            '<section style="margin:0 0 14px;padding:13px 16px 16px;'
            f'border:1px solid #e0e4e8;border-left:6px solid {escape(color, quote=True)};'
            'border-radius:9px;background:#fff;">'
            '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
            'margin-bottom:10px;font:600 14px system-ui,sans-serif;color:#243041;">'
            f'<span style="display:inline-block;width:13px;height:13px;border-radius:3px;'
            f'background:{escape(color, quote=True)};"></span>'
            f'Fragment {step.number} · {escape(atoms)} · +{len(step.new_bytes)} bytes'
            f'<span style="font-weight:400;color:#657080;">{escape(note)}</span>'
            '</div>'
            '<div style="padding:13px 14px;border:1px solid #e7eaee;border-radius:6px;'
            'background:#fafbfc;line-height:2.0;">'
            '<code style="font:600 16px/2.0 ui-monospace,SFMono-Regular,Menlo,monospace;'
            'letter-spacing:.025em;color:#17212e;white-space:normal;word-break:break-all;">'
            + tag + '</code></div></section>'
        )
    display(HTML(
        '<div style="max-width:1040px;min-width:0;">'
        '<div style="font:600 18px system-ui,sans-serif;color:#182230;'
        'margin:0 0 5px;">Conformer tag by fragment</div>'
        + (
            '<div style="padding:13px 14px;margin:10px 0 15px;'
            'border:1px solid #e7eaee;border-radius:6px;background:#fafbfc;">'
            '<div style="font:600 13px system-ui,sans-serif;color:#657080;'
            'margin-bottom:6px;">Base SMILES</div>'
            '<code style="font:600 16px/1.8 ui-monospace,SFMono-Regular,Menlo,monospace;'
            'word-break:break-all;">'
            + _colored_smiles(base_smiles, steps, colors) + '</code></div>'
            if base_smiles else ''
        )
        +
        '<div style="font:13px system-ui,sans-serif;color:#657080;margin:0 0 16px;">'
        'Each row contains the full tag so far. The newest bytes are underlined; '
        'a split glyph contains bits from two fragments.</div>'
        + ''.join(cards) + '</div>'
    ))
