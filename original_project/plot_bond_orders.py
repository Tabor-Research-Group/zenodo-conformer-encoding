"""Highlight a Psience molecule's bonds by RDKit bond order."""

from rdkit import Chem
from Psience.Molecools import Molecule


def plot_bonds_by_order(
    molecule: Molecule,
    *,
    include_hydrogens: bool = False,
    single_color: str = '#85BADD',
    double_color: str = '#EF7F70',
    aromatic_color: str = '#B985BD',
    triple_color: str = 'green',
    other_color: str = 'orange',
    image_size: int = 800,
    bond_style=None,
    atom_style=None,
    **plot_options,
):
    """Draw the molecule with every bond highlighted by bond type.

    Single and double bonds use their respective colors. Aromatic, triple,
    and other bond types use ``other_color``; aromatic bonds are not assigned
    an arbitrary Kekulé single/double pattern.

    ``include_hydrogens`` controls explicit H atoms already in ``molecule``.
    It does not add implicit hydrogens to a molecule that lacks them. The
    input molecule is not modified. Extra options are forwarded to
    ``molecule.plot``.
    """
    render_molecule = molecule if include_hydrogens else molecule.remove_hydrogens()
    rdkit_molecule = render_molecule.rdmol.mol.GetOwningMol()

    if atom_style is None:
        atom_style = {}
    if bond_style is None:
        bond_style = {}
    atom_bond_type_map = {}
    for bond in rdkit_molecule.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if (j,i) in bond_style:
            i,j = j,i
        bo = 0
        if bond.GetBondType() == Chem.BondType.SINGLE:
            bo = 1
            color = single_color
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            bo = 2
            color = double_color
        elif bond.GetBondType() == Chem.BondType.AROMATIC:
            bo = 2.5
            color = aromatic_color
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            bo = 3
            color = triple_color
        else:
            color = other_color
        atom_bond_type_map[i] = max(atom_bond_type_map.get(i, 0), bo)
        atom_bond_type_map[j] = max(atom_bond_type_map.get(j, 0), bo)
        bond_style[(i,j)] = {'glow': color} | bond_style.get((i,j), {})

    for i,bo in atom_bond_type_map.items():
        if bo == 1:
            color = single_color
        elif bo == 2:
            color = double_color
        elif bo == 2.5:
            color = aromatic_color
        elif bo == 3:
            color = triple_color
        else:
            color = other_color
        atom_style[i] = {'glow': color} | atom_style.get(i, {})

    options = dict(
        backend='rdkit',
        format='svg',
        image_size=image_size,
        # Hydrogens were already selected above. Suppressing them again in
        # the drawer would change bond indices and discard H-bond highlights.
        remove_hydrogens=False,
        draw_implicit_hydrogens=False,
        highlight_atoms=[],
        highlight_bonds=list(bond_style),
        bond_style=bond_style,
        atom_style=atom_style,
    ) | plot_options
    options['backend'] = 'rdkit'
    options['remove_hydrogens'] = False
    return render_molecule.plot(**options)
