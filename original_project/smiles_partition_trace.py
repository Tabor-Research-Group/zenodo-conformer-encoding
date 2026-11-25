"""Track how a conformer SMILES tag changes as partition blocks are added.

Notebook use::

    from smiles_partition_trace import track_smi_partition, show_smi_changes
    steps = track_smi_partition(smiles)
    show_smi_changes(steps)

``block`` and ``atom_order`` use the heavy-atom indices of the full canonical
SMILES after its first Psience round trip. The displayed diff is for the
base64 conformer tag, not the chemical SMILES before the underscore.
"""

import base64
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import escape

import numpy as np
from McUtils.ExternalPrograms import RDMolecule
from Psience.Molecools import Molecule


def make_smi(smi, include_tag=True, sel=None):
    """The supplied canonicalization and submolecule path."""
    mol = Molecule.from_string(
        smi, 'smi', confgen_opts={'random_seed': 254}
    )
    smi = mol.to_string(
        'smi', include_tag=include_tag, canonical=True,
        remove_hydrogens=True,
    )
    if sel is not None:
        mol = Molecule.from_string(smi, 'smi')
        mol = mol.take_submolecule(sel)
        smi = mol.to_string(
            'smi', include_tag=include_tag, canonical=True,
            remove_hydrogens=True,
        )
    return smi


def _split_tag(smi, include_tag):
    if not include_tag:
        return smi, None
    core, separator, tag = smi.partition('_')
    if not separator:
        raise ValueError('Expected a conformer tag after the SMILES')
    return core, tag


def _tag_diff(old, new):
    """Base64 character edits: operation, old substring, new substring."""
    matcher = SequenceMatcher(None, old, new, autojunk=False)
    return tuple(
        (op, old[i:j], new[k:l])
        for op, i, j, k, l in matcher.get_opcodes()
    )


def _tag_words(tag):
    """Read the native packed 16-bit values underlying McUtils's base64 tag."""
    if tag is None:
        return ()
    if RDMolecule.default_tag_byte_size != 16:
        raise NotImplementedError('This tracker expects the default 16-bit tag encoding')
    data = base64.b64decode(tag, validate=True)
    if len(data) % 2:
        raise ValueError('Conformer tag contains an incomplete 16-bit word')
    return tuple(int(value) for value in np.frombuffer(data, dtype=np.uint16))


def _words_by_atom(atom_order, words):
    """Map packed words to the atom positions used by the canonical Z-matrix."""
    if len(words) != 2 * len(atom_order) - 3:
        raise ValueError(
            f'Expected {2 * len(atom_order) - 3} packed values for '
            f'{len(atom_order)} atoms, found {len(words)}'
        )
    result = {}
    for position, atom in enumerate(atom_order):
        if position == 0:
            result[atom] = {}
        elif position == 1:
            result[atom] = {'distance': words[0]}
        elif position == 2:
            result[atom] = {'distance': words[1], 'angle': words[2]}
        else:
            result[atom] = {
                'distance': words[2 * position - 3],
                'angle+dihedral': words[2 * position - 2],
            }
    return result


@dataclass(frozen=True)
class TagWordChange:
    atom: int
    coordinate: str
    before: int | None
    after: int | None


def _word_changes(previous_order, previous_words, order, words):
    before = _words_by_atom(previous_order, previous_words) if previous_order else {}
    after = _words_by_atom(order, words)
    changes = []
    for atom in order:
        old_values = before.get(atom, {})
        new_values = after[atom]
        for coordinate in dict.fromkeys((*old_values, *new_values)):
            old_value = old_values.get(coordinate)
            new_value = new_values.get(coordinate)
            if old_value != new_value:
                changes.append(TagWordChange(atom, coordinate, old_value, new_value))
    return tuple(changes)


@dataclass(frozen=True)
class SmilesStep:
    number: int
    block: tuple[int, ...]
    selected: tuple[int, ...]
    smiles: str
    core: str
    tag: str | None
    atom_order: tuple[int, ...]
    tag_edits: tuple[tuple[str, str, str], ...]
    tag_words: tuple[int, ...]
    changed_words: tuple[TagWordChange, ...]
    full_core: str
    full_tag: str | None


def track_smi_partition(smi, *, include_tag=True, blocks=None):
    """Return one step for each accumulated canonical fragment.

    ``blocks`` may override the default partition. Each block is a sequence of
    heavy-atom indices in the full canonical SMILES, in the desired order.
    ``atom_order`` maps atoms in each output SMILES back to those same indices.
    """
    if not include_tag:
        raise ValueError('Tag tracking requires include_tag=True')
    full_smi = make_smi(smi, include_tag=include_tag)
    full_core, full_tag = _split_tag(full_smi, include_tag)
    base = Molecule.from_string(full_smi, 'smi')
    heavy_positions = tuple(
        i for i, atom in enumerate(base.atoms) if atom not in {'H', 'D', 'T'}
    )
    heavy = base.remove_hydrogens()
    if blocks is None:
        fragments = heavy.edge_graph.get_canonical_fragments(
            np.arange(len(heavy.atoms))
        )
        blocks = [fragment for _, fragment in fragments]

    steps = []
    selected_heavy = []
    previous_tag = ''
    previous_order = ()
    previous_words = ()
    for number, block in enumerate(blocks, start=1):
        block = tuple(int(i) for i in block)
        if any(i < 0 or i >= len(heavy_positions) for i in block):
            raise IndexError(f'Block {number} contains an invalid heavy-atom index')
        if any(i in selected_heavy for i in block):
            raise ValueError(f'Block {number} repeats an earlier atom')
        selected_heavy.extend(block)
        selected = tuple(heavy_positions[i] for i in selected_heavy)
        fragment = base.take_submolecule(selected)
        output, local_order = fragment.to_string(
            'smi', include_tag=include_tag, canonical=True,
            remove_hydrogens=True, return_reordering=True,
        )
        core, tag = _split_tag(output, include_tag)
        atom_order = tuple(selected[int(i)] for i in local_order)
        words = _tag_words(tag)
        steps.append(SmilesStep(
            number, block, tuple(selected_heavy), output, core, tag,
            atom_order, _tag_diff(previous_tag, tag or ''), words,
            _word_changes(previous_order, previous_words, atom_order, words)
                if tag is not None else (),
            full_core, full_tag,
        ))
        previous_tag = tag or ''
        previous_order = atom_order
        previous_words = words
    return steps


def show_smi_changes(steps, *, show_atom_order=False, show_tag=True):
    """Highlight changes to the base64 tag and its packed values in Jupyter."""
    from IPython.display import HTML, display

    rows = []
    for step in steps:
        old_markup = []
        new_markup = []
        for operation, old, new in step.tag_edits:
            if operation == 'equal':
                old_markup.append(escape(old))
                new_markup.append(escape(new))
            else:
                if old:
                    old_markup.append(
                        '<mark style="background:#ffd5d5">' + escape(old) + '</mark>'
                    )
                if new:
                    new_markup.append(
                        '<mark style="background:#d8f4d2">' + escape(new) + '</mark>'
                    )
        old_display = ''.join(old_markup) if step.number > 1 else '∅'
        order = (
            f'<br><small>SMILES atom order: {escape(str(step.atom_order))}</small>'
            if show_atom_order else ''
        )
        word_changes = '<br>'.join(
            f'{change.atom} {escape(change.coordinate)}: '
            f'{"—" if change.before is None else f"{change.before:04X}"}'
            f' → {"—" if change.after is None else f"{change.after:04X}"}'
            for change in step.changed_words
        ) or 'none'
        rows.append(
            '<tr><td>' + str(step.number) + '</td><td>'
            + escape(str(step.block)) + '</td><td><code style="white-space:pre-wrap;word-break:break-all">'
            + (old_display if show_tag else '')
            + '</code></td><td><code style="white-space:pre-wrap;word-break:break-all">'
            + (''.join(new_markup) if show_tag else '') + '</code>'
            + order + '</td><td><small>' + word_changes + '</small></td></tr>'
        )

    table = (
        '<table style="border-collapse:collapse">'
        '<thead><tr><th>Step</th><th>New atom indices</th>'
        '<th>Previous tag</th><th>Current tag</th>'
        '<th>Changed packed values (hex)</th></tr></thead>'
        '<tbody>' + ''.join(rows) + '</tbody></table>'
    )
    if steps and steps[-1].tag != steps[-1].full_tag:
        table += (
            '<p><strong>The full-selection tag differs from the unselected '
            'tag.</strong><br>Unselected: <code>'
            + escape(steps[-1].full_tag or '') + '</code><br>Full selection: <code>'
            + escape(steps[-1].tag or '') + '</code></p>'
        )
    display(HTML(table))
