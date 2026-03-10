# MeXpose v2.0 Extension: Single-Cell Intra- vs Extracellular Signal Analysis

This notebook extends the MeXpose v2.0 single-cell workflow by enabling spatial comparison of intracellular and extracellular metal signal distributions in LA-ICP-TOFMS imaging data.

Using reconstructed elemental TIFF images together with segmentation masks, the workflow allows users to define spatial regions of interest (ROIs) interactively and summarize signal contributions originating from intracellular and extracellular compartments.

## Workflow

1. Load reconstructed elemental TIFF image and corresponding segmentation mask.
2. Define ROIs interactively using lasso or rectangular selection.
3. Convert the ROI polygon into a pixel-level mask.
4. Classify pixels as:
   - **Intracellular** (`label > 0`)
   - **Extracellular** (`label = 0`)
5. Compute ROI statistics from pixel intensities.

## Inputs

- reconstructed elemental TIFF image (e.g. Pt signal)
- cell segmentation mask with integer labels
- optional single-cell feature table for visualization

## Outputs

For each ROI the workflow exports:

- **Per-cell CSV (`<name>.csv`)**  
  Subset of the single-cell table for selected cells with ROI tags.

- **ROI statistics CSV (`<name>_intra_extra_stats.csv`)**  
  Summary statistics including intracellular/extracellular pixel counts, summed signal intensities, and signal ratios.

- **ROI figure (`<name>_figure.png`)**  
  Cropped visualization of the ROI with intracellular/extracellular overlay.

The interactive Plotly viewer also allows exporting the current view via the figure toolbar.

## Notes

Pixel values represent **counts** and are not converted to concentration units.

This notebook is an **extension to the MeXpose v2 workflow** and uses the same environment and dependencies.

More detailed explanations of the workflow, implementation details, and methodological notes are provided in the **Markdown documentation within the notebook itself**.
