"""Draw a canonical fragment partition from a SMILES string.

Set PYTHONPATH to the McUtils and Psience checkout roots. The default
plot uses the McUtils atom-label placement patch; --baseline uses
RDKit's existing display_atom_numbers=True for a direct placement reference.

In a notebook, call prepare_partition(smiles), draw_partition(), and optionally
save_partition(). The drawn atom order follows the reported canonical SMILES.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import inspect
import json
import platform
import re
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D
import McUtils
import Psience
import McUtils.Plots as plt
from McUtils.ExternalPrograms import RDMolecule
from Psience.Molecools import Molecule


OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_ROOTS = {
    'McUtils': Path('/Users/Mark/Documents/UW/Research/Development/McUtils'),
    'Psience': Path('/Users/Mark/Documents/UW/Research/Development/Psience'),
}
SMILES = (
    'CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)'
    'C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4'
)
PALETTE = (
    plt.prep_color('green', lighten=.5),
    plt.prep_color('purple', lighten=.8),
    plt.prep_color('orange', lighten=.2),
    'teal',
    plt.prep_color('red', lighten=.5),
    plt.prep_color('blue', lighten=.8),
)
SINGLE_ATOM_PALETTE = ('#85BADD',)
# Kept for notebooks that use the old scalar default.
SINGLE_ATOM_COLOR = SINGLE_ATOM_PALETTE[0]

# Extra distance, in molecule 2D coordinates, along RDKit's chosen direction
# from each atom to its number. Set to zero for RDKit's original positions.
DEFAULT_ATOM_LABEL_OFFSET = 0.2

# Edit these dictionaries to move labels by hand. Atom-number offsets are
# added to DEFAULT_ATOM_LABEL_OFFSET; position is an absolute label center.
# Both use molecule 2D coordinates, with positive y above the atom. Scale
# multiplies the native number size. Indices follow the canonical SMILES.
ATOM_LABEL_ADJUSTMENTS: dict[int, dict] = {
    # 0: {'offset': (0.0, 0.2)},
    # 2: {'scale': 1.15},
}

# Colors for the opt-in r1/r2/a1 annotations. Edit these in a notebook,
# or pass draw_coords_palette to draw_partition for one drawing.
DRAW_COORDS_PALETTE = {
    'r1': '#D4232F',
    'r2': '#1F54DC',
    'a1': '#168B32',
}

# Raw RDMolecule draw_coords options for each built-in annotation. Change any
# supported field here, including radius, scaling, key, styles, and label.
# A label dict is merged with the generated label; other fields replace defaults.
DRAW_COORDS_OPTIONS: dict[str, dict] = {
    'r1': {},
    'r2': {},
    'a1': {'radius': .47, 'scaling': .8},
}

# The r1/r2/a1 offsets are relative to the defaults in make_draw_coords;
# position replaces the default with an absolute molecule 2D coordinate.
DRAW_LABEL_ADJUSTMENTS: dict[str, dict] = {
    # 'r1': {'offset': (0.0, 0.2)},
    # 'a1': {'font_size': 36},
}


@dataclass
class PartitionData:
    molecule: Molecule
    rd_mol: Chem.Mol
    fragments: list
    atom_colors: dict[int, str]
    input_smiles: str
    canonical_smiles: str
    palette: tuple[str, ...] = PALETTE
    single_atom_palette: tuple[str, ...] = SINGLE_ATOM_PALETTE


def verify_sources(*, require_label_patch):
    if platform.machine() != 'arm64':
        raise RuntimeError('Use the native ARM Python 3.11 environment')
    for name, module in (('McUtils', McUtils), ('Psience', Psience)):
        package_file = Path(module.__file__).resolve()
        if not package_file.is_relative_to(SOURCE_ROOTS[name]):
            raise RuntimeError(f'{name} resolved to {package_file}, not its source checkout')
    if require_label_patch and 'atom_label_placement' not in inspect.signature(RDMolecule.draw).parameters:
        raise RuntimeError('Apply McUtils-on-disk-source-rebuilt.patch before drawing colored atom labels')


def make_molecule(smiles=None):
    """Canonicalize with Psience, then draw that exact SMILES atom order."""
    if smiles is None:
        smiles = SMILES
    canonical_smiles = Molecule.from_string(smiles, 'smi').to_string(
        'smi', canonical=True, remove_hydrogens=True
    )
    rd_mol = Chem.MolFromSmiles(canonical_smiles)
    if rd_mol is None:
        raise ValueError('Canonical SMILES could not be parsed by RDKit')
    rdDepictor.Compute2DCoords(rd_mol)
    # Psience retains bond orders but not aromatic nitrogen's explicit H count.
    Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    mol = Molecule.from_rdmol(RDMolecule(rd_mol.GetConformer()))
    drawn = mol.remove_hydrogens().rdmol.mol.GetOwningMol()

    def topology(m):
        atoms = [atom.GetSymbol() for atom in m.GetAtoms()]
        bonds = sorted(
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
             max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
             str(b.GetBondType()))
            for b in m.GetBonds()
        )
        return atoms, bonds

    if topology(drawn) != topology(rd_mol):
        raise RuntimeError('Drawn atom order does not match the canonical SMILES')
    return mol, rd_mol, canonical_smiles


def _palette_colors(value, default, name):
    if value is None:
        value = default
    if isinstance(value, str):
        value = (value,)
    colors = tuple(value)
    if not colors:
        raise ValueError(f'{name} must contain at least one color')
    return colors


def fragment_styles(mol, *, palette=None, single_atom_palette=None):
    """Color multi-atom and single-atom fragments with separate cycles."""
    palette = _palette_colors(palette, PALETTE, 'palette')
    single_atom_palette = _palette_colors(
        single_atom_palette, SINGLE_ATOM_PALETTE, 'single_atom_palette'
    )
    fragments = mol.edge_graph.get_canonical_fragments(np.arange(len(mol.atoms)))
    assert sum(len(indices) for _, indices in fragments) == len(mol.atoms)
    styles = {}
    colors = {}
    palette_index = 0
    single_atom_index = 0
    for _, indices in fragments:
        if len(indices) == 1:
            color = single_atom_palette[single_atom_index % len(single_atom_palette)]
            single_atom_index += 1
        else:
            color = palette[palette_index % len(palette)]
            palette_index += 1
        for index in indices:
            styles[int(index)] = {'glow': color}
            colors[int(index)] = color
    return fragments, styles, colors


def prepare_partition(smiles=None, *, palette=None, single_atom_palette=None):
    """Build fragments in canonical SMILES order with independent color cycles."""
    verify_sources(require_label_patch=False)
    if smiles is None:
        smiles = SMILES
    palette = _palette_colors(palette, PALETTE, 'palette')
    single_atom_palette = _palette_colors(
        single_atom_palette, SINGLE_ATOM_PALETTE, 'single_atom_palette'
    )
    molecule, rd_mol, canonical_smiles = make_molecule(smiles)
    fragments, _, colors = fragment_styles(
        molecule, palette=palette, single_atom_palette=single_atom_palette
    )
    return PartitionData(molecule, rd_mol, fragments, colors,
                         smiles, canonical_smiles, palette, single_atom_palette)


def plot_chain_decomp(mol, *, palette=None, single_atom_palette=None, **etc):
    """Color canonical fragments and draw with the RDKit backend.

    ``single_atom_palette`` advances once per singleton fragment. Multi-atom
    fragments advance ``palette`` without consuming singleton colors.
    """
    heavy = mol.remove_hydrogens()
    _, styles, _ = fragment_styles(
        heavy, palette=palette, single_atom_palette=single_atom_palette
    )
    options = dict(
        backend='rdkit', image_size=800, use_coords=True,
        bond_radius=.4, atom_radii=.33,
        atom_style=styles, atom_highlights_are_circles=True,
        blend_mixed_bonds=False,
    )
    return heavy.plot(**(options | etc))


def make_draw_coords(xy, label_adjustments=None, *, palette=None,
                     options=None, extra=None):
    """Build built-in annotations, then append any RDMolecule draw_coords specs.

    ``options`` overrides entries in DRAW_COORDS_OPTIONS by r1/r2/a1 name.
    ``extra`` accepts the drawer's dict, list-of-keys, or list-of-dicts forms.
    """
    if label_adjustments is None:
        label_adjustments = DRAW_LABEL_ADJUSTMENTS
    if len(xy) < 4:
        raise ValueError('The opt-in r1/r2/a1 annotations need at least four atoms')
    palette = DRAW_COORDS_PALETTE | ({} if palette is None else palette)
    if options is None:
        options = {}

    def merge_annotation(base, changes):
        merged = base | changes
        if isinstance(base.get('label'), dict) and isinstance(changes.get('label'), dict):
            merged['label'] = base['label'] | changes['label']
        return merged

    def label_style(text, color, position, *, display_text=None):
        style = {'text': text if display_text is None else display_text,
                 'color': color, 'font_size': 32}
        adjustments = label_adjustments.get(text, {})
        position = np.asarray(adjustments.get('position', position), dtype=float)
        offset = np.asarray(adjustments.get('offset', (0, 0)), dtype=float)
        if position.shape != (2,) or offset.shape != (2,):
            raise ValueError(f'{text} position and offset must each have two components')
        style['position'] = position + offset
        style.update({k: v for k, v in adjustments.items() if k not in ('position', 'offset')})
        return style

    def bond_label(i, j, text, color, position, *, display_text=None):
        return {
            'key': (i, j), 'color': None,
            'label': label_style(text, color, position, display_text=display_text),
        }
    builtins = {
        'r1': bond_label(0, 1, 'r1', palette['r1'], xy[0] + [-.8, 1.0],
                         display_text='r<sub>1</sub>'),
        'r2': bond_label(1, 2, 'r2', palette['r2'], (xy[1] + xy[2]) / 2 + [-.18, -1.25],
                         display_text='r<sub>2</sub>'),
        'a1': {
            'key': (1, 2, 3), 'color': palette['a1'],
            'label': label_style('a1', palette['a1'], xy[2] + [.53, -1.25],
                                 display_text='a<sub>1</sub>'),
        },
    }
    annotations = []
    for name, spec in builtins.items():
        spec = merge_annotation(spec, DRAW_COORDS_OPTIONS.get(name, {}))
        annotations.append(merge_annotation(spec, options.get(name, {})))
    return annotations + deepcopy(RDMolecule._prep_draw_coords(extra))


def make_default_atom_label_offsets(partition, distance=None,
                                    *, image_size=800, draw_label_adjustments=None):
    """Move each atom number away from its atom along RDKit's native direction.

    The scalar distance uses molecule 2D coordinates. A zero distance keeps
    RDKit's exact atom-number positions and skips the reference drawing.
    """
    if distance is None:
        distance = DEFAULT_ATOM_LABEL_OFFSET
    distance = float(distance)
    if not np.isfinite(distance):
        raise ValueError('default atom-label offset must be finite')
    if distance == 0:
        return {}

    reference = draw_partition(partition, baseline=True, image_size=image_size)
    note_paths = re.findall(
        r"<path class='note' d='(.*?)' fill='#[0-9A-Fa-f]{6}'/>",
        reference.text, re.S,
    )
    expected = sum(len(str(i)) for i in partition.atom_colors)
    if len(note_paths) != expected:
        raise ValueError(
            f'RDKit drew {len(note_paths)} atom-note glyphs, expected {expected}'
        )

    number_pattern = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
    drawer = reference.figure
    xy = partition.rd_mol.GetConformer().GetPositions()[:, :2]
    offsets = {}
    next_path = 0
    for i in sorted(partition.atom_colors):
        glyph_count = len(str(i))
        glyphs = note_paths[next_path:next_path + glyph_count]
        next_path += glyph_count
        points = []
        for glyph in glyphs:
            values = [float(value) for value in number_pattern.findall(glyph)]
            if len(values) % 2:
                raise ValueError(f'cannot locate RDKit note glyph for atom {i}')
            points.extend(np.asarray(values).reshape(-1, 2))
        points = np.asarray(points)
        label_center = (points.min(axis=0) + points.max(axis=0)) / 2

        x, y = map(float, xy[i])
        origin = drawer.GetDrawCoords(Point2D(x, y))
        pixel_x = drawer.GetDrawCoords(Point2D(x + 1, y))
        pixel_y = drawer.GetDrawCoords(Point2D(x, y + 1))
        origin = np.array([origin.x, origin.y])
        basis = np.array([
            [pixel_x.x - origin[0], pixel_y.x - origin[0]],
            [pixel_x.y - origin[1], pixel_y.y - origin[1]],
        ])
        direction = np.linalg.solve(basis, label_center - origin)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(f'RDKit placed atom label {i} at its atom center')
        offsets[i] = tuple(distance * direction / norm)
    return offsets


def make_atom_labels(partition, *, default_offset=None,
                     adjustments=None, image_size=800,
                     draw_label_adjustments=None):
    """Combine the common outward offset with per-atom manual adjustments."""
    if adjustments is None:
        adjustments = ATOM_LABEL_ADJUSTMENTS
    default_offsets = make_default_atom_label_offsets(
        partition, default_offset, image_size=image_size,
        draw_label_adjustments=draw_label_adjustments,
    )
    labels = {}
    for i, color in partition.atom_colors.items():
        adjustment = adjustments.get(i, {})
        base_offset = (0, 0) if 'position' in adjustment else default_offsets.get(i, (0, 0))
        manual_offset = np.asarray(adjustment.get('offset', (0, 0)), dtype=float)
        if manual_offset.shape != (2,):
            raise ValueError(f'atom-label offset for {i} must have two components')
        offset = np.asarray(base_offset) + manual_offset
        label = {'text': str(i), 'color': color}
        label.update({k: v for k, v in adjustment.items() if k != 'offset'})
        if np.any(offset):
            label['offset'] = tuple(offset)
        labels[i] = label
    return labels


def draw_partition(partition: PartitionData | str | None = None, *, smiles=None, baseline=False,
                   atom_label_adjustments=None, draw_label_adjustments=None,
                   palette=None, single_atom_palette=None,
                   draw_coords_palette=None, draw_coords_options=None,
                   extra_draw_coords=None, include_default_draw_coords=False,
                   default_atom_label_offset=None,
                   image_size=800, **plot_options):
    """Return a notebook-displayable figure; no files are written.

    The first argument may be a prepared partition or a SMILES string. Default
    coordinates are molecule-specific, so they are opt-in. Extra
    keyword arguments override generated options passed to mol.plot();
    ``draw_coords`` replaces all generated coordinate annotations if supplied.
    ``palette`` and ``single_atom_palette`` can override a prepared partition's
    colors for this drawing, including its atom-number labels.
    """
    verify_sources(require_label_patch=not baseline)
    if isinstance(partition, str):
        if smiles is not None:
            raise ValueError('Pass the SMILES either positionally or by keyword, not both')
        smiles, partition = partition, None
    if partition is None:
        partition = prepare_partition(
            smiles, palette=palette, single_atom_palette=single_atom_palette
        )
    elif smiles is not None:
        raise ValueError('Pass either a prepared partition or smiles, not both')
    elif not isinstance(partition, PartitionData):
        raise TypeError('The first argument must be a SMILES string or PartitionData')
    if palette is not None or single_atom_palette is not None:
        palette = _palette_colors(palette, partition.palette, 'palette')
        single_atom_palette = _palette_colors(
            single_atom_palette, partition.single_atom_palette,
            'single_atom_palette'
        )
        _, _, colors = fragment_styles(
            partition.molecule.remove_hydrogens(), palette=palette,
            single_atom_palette=single_atom_palette
        )
        partition = replace(
            partition, atom_colors=colors, palette=palette,
            single_atom_palette=single_atom_palette
        )
    options = {
        'format': 'svg',
        'image_size': image_size,
    }
    use_default_coords = include_default_draw_coords or any(
        value is not None for value in (
            draw_label_adjustments, draw_coords_palette, draw_coords_options
        )
    )
    if 'draw_coords' not in plot_options:
        if use_default_coords:
            xy = partition.rd_mol.GetConformer().GetPositions()[:, :2]
            options['draw_coords'] = make_draw_coords(
                xy, draw_label_adjustments, palette=draw_coords_palette,
                options=draw_coords_options, extra=extra_draw_coords
            )
        elif extra_draw_coords is not None:
            options['draw_coords'] = deepcopy(RDMolecule._prep_draw_coords(extra_draw_coords))
    if baseline:
        options['display_atom_numbers'] = True
    else:
        if 'atom_labels' not in plot_options:
            options['atom_labels'] = make_atom_labels(
                partition, default_offset=default_atom_label_offset,
                adjustments=atom_label_adjustments, image_size=image_size,
                draw_label_adjustments=draw_label_adjustments,
            )
        options['atom_label_placement'] = 'rdkit'
    return plot_chain_decomp(
        partition.molecule, palette=partition.palette,
        single_atom_palette=partition.single_atom_palette,
        **(options | plot_options)
    )


def save_png(svg_path, png_path):
    import ctypes.util
    arm_cairo = Path('/opt/homebrew/lib/libcairo.2.dylib')
    old_find_library = ctypes.util.find_library
    if arm_cairo.exists():
        ctypes.util.find_library = lambda name: (
            str(arm_cairo) if 'cairo' in name else old_find_library(name)
        )
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    finally:
        ctypes.util.find_library = old_find_library


def save_partition(figure, partition, *, baseline=False, output_dir=OUTPUT_DIR,
                   stem='canonical_partition', make_png=True):
    """Save an existing figure and its partition metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline:
        stem += '_baseline'
    svg_path = output_dir / f'{stem}.svg'
    png_path = output_dir / f'{stem}.png'
    metadata_path = output_dir / f'{stem}.json'
    figure.save(svg_path)
    if make_png:
        save_png(svg_path, png_path)
    metadata_path.write_text(json.dumps({
        'input_smiles': partition.input_smiles,
        'canonical_smiles': partition.canonical_smiles,
        'heavy_atom_symbols': [atom.GetSymbol() for atom in partition.rd_mol.GetAtoms()],
        'fragment_atom_indices': [
            [int(i) for i in indices] for _, indices in partition.fragments
        ],
    }, indent=2) + '\n')
    return {'svg': svg_path, 'png': png_path if make_png else None,
            'metadata': metadata_path}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', default=None, help='SMILES to canonicalize and plot')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args(argv)
    partition = prepare_partition(args.smiles)
    figure = draw_partition(partition, baseline=args.baseline)
    paths = save_partition(figure, partition, baseline=args.baseline,
                           output_dir=args.output_dir)
    print(f'Canonical SMILES: {partition.canonical_smiles}')
    print(f'{len(partition.fragments)} fragments; saved {paths["svg"]} and {paths["png"]}')
    return figure


if __name__ == '__main__':
    main()
