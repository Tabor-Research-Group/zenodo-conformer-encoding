// Build a PPTX with native, editable colored SMILES and tag text boxes.
// Called by export_tag_prefixes_pptx in smiles_tag_prefix_trace.py.
import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  throw new Error('Usage: tag_prefix_to_pptx.mjs INPUT.json OUTPUT.pptx');
}
const { slides } = JSON.parse(await fs.readFile(input, 'utf8'));
if (!Array.isArray(slides) || !slides.length) {
  throw new Error('The input must contain at least one slide');
}

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});
for (const item of slides) {
  if (item.smiles_runs.map(run => run.text).join('') !== item.base_smiles) {
    throw new Error('Colored runs do not reconstruct the base SMILES');
  }
  if (item.runs.map(run => run.text).join('') !== item.tag) {
    throw new Error('Colored runs do not reconstruct the tag');
  }
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';

  const title = slide.shapes.add({
    geometry: 'textbox',
    position: { left: 60, top: 50, width: 1160, height: 55 },
    fill: 'none',
    line: { fill: 'none', width: 0 },
  });
  title.text = item.title;
  title.text.style = {
    typeface: 'Arial', fontSize: 28, bold: true,
    color: '#243041', autoFit: 'none',
  };

  function addLabel(text, top) {
    const label = slide.shapes.add({
      geometry: 'textbox',
      position: { left: 60, top, width: 1160, height: 32 },
      fill: 'none', line: { fill: 'none', width: 0 },
    });
    label.text = text;
    label.text.style = {
      typeface: 'Arial', fontSize: 17, bold: true,
      color: '#657080', autoFit: 'none',
    };
  }
  function addColoredBox(runs, top, height) {
    const box = slide.shapes.add({
      geometry: 'textbox',
      position: { left: 60, top, width: 1160, height },
      fill: 'none',
      line: { fill: 'none', width: 0 },
    });
    box.text.style = {
      typeface: 'Courier New', fontSize: 28, color: '#243041',
      autoFit: 'none',
    };
    box.text = [{
      runs: runs.map(({ text, color, new: isNew }) => ({
        run: text,
        textStyle: {
          typeface: 'Courier New', fontSize: '21pt', color,
          bold: true, ...(isNew ? { underline: 'sng' } : {}),
        },
      })),
    }];
    return box;
  }

  addLabel('Base SMILES', 130);
  addColoredBox(item.smiles_runs, 169, 165);
  addLabel('Conformer tag', 365);
  addColoredBox(item.runs, 403, 250);
  slide.speakerNotes.textFrame.setText(
    'Copy either colored text box into your presentation. All characters '
    + 'and their colors are editable. Line wrapping is visual only. '
    + 'Opening branch parentheses and bond symbols follow the next atom; '
    + 'ambiguous punctuation follows the preceding atom.'
  );
}
await (await PresentationFile.exportPptx(presentation)).save(output);
