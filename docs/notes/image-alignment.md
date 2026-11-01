# Splat-guided photograph alignment

`spirula align-images` is a separate preprocessing command using the existing
dataset readers, splat renderer, native RoMa v2 and image I/O. It does not train
Gaussians or modify the editor. The source dataset is read-only; output must be
a fresh directory outside it, including through symlink aliases.

```sh
spirula align-images --data DATASET --reference MODEL.ply --output REVIEW \
  --checkpoint CHECKPOINT.pt --sample 10 --seed 20261031 --keep-alignment-data
```

The default is a uniform sample of ten registered images without replacement.
The selected source views are never replaced with easier photographs. With
`--keep-alignment-data`, `selection.json` records them before rendering and the
review includes failures and copies of sampled originals for native inspection.

Full conversion uses the same path and requires explicit selection:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output CORRECTED \
  --checkpoint CHECKPOINT.pt --all
```

`--all` does not copy originals into the report. No full-dataset conversion is
needed for evaluation. Without `--max-resolution`, rendering and fine matching
use each photograph’s native dimensions with no resolution cap. Providing
`--max-resolution PIXELS` explicitly caps the longest rendering/matching side;
delivered images retain native dimensions in both cases. `--primitive
3dgs|mip|3dgut` selects the renderer. The reference must share the dataset's
coordinate frame. Placement and appearance corrections are not inferred from
a loose PLY. Exact supported camera models are retained; approximate lens fits
are refused before output creation. Matching defaults to FP32; `--precision
mixed|auto` selects the existing inference alternatives.

## Minimal training output

The default writes only corrected `images/` and lossless `masks/`. No JSON files,
camera metadata, point clouds, alignment caches, warp fields, diagnostic images,
review page or original photographs are copied or generated. Use the original
dataset's camera alignment and existing point cloud for training. The corrected
output is an image/mask replacement, not a standalone reconstructed dataset.

The default pipeline prefetches one input and overlaps CPU fitting, native
resampling, quality checks and encoding with the next view's GPU rendering and
matching. Only one view uses the GPU at a time; renderer and matcher allocations
remain separate. One CPU output job and one prefetched input bound the queue in
RAM independently of dataset size, with no temporary render or matching spool
on disk. The CPU output worker uses the OpenMP thread budget minus two (minimum
one), leaving room for input preparation and GPU submission. Set
`OMP_NUM_THREADS` to control that budget. `--no-pipeline` restores serial
execution and reduces peak host memory. It does not change correction settings
or retained output files.

On three seeded CinqueTorri views with 1280-pixel working resolution, native JPEG
quality 95 output, eight OpenMP threads and an RTX 3090, elapsed time was 40.24 s
serial and 32.46 s pipelined. All images and masks were byte-identical.
This small comparison includes startup and is not a throughput
forecast for native-resolution matching or a full dataset.

`--reuse-renders RUN`
can read compatible renders and matches directly from an existing alignment
folder without copying those caches into the output.

Add `--keep-alignment-data` to retain the review, diagnostics and reusable
caches, including diagnostic JSON files. Camera metadata and point clouds are
never exported. `--resume` and `--render-only` require this option.

## Output encoding

Corrected images default to lossless PNG. Select JPEG and its quality explicitly:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output CORRECTED \
  --image-format jpeg --jpeg-quality 95 --all
```

`--image-format` accepts `png`, `jpeg` and the alias `jpg`. JPEG quality is an
integer from 1 to 100, default 95, and applies only to JPEG output. Masks and
review diagnostics always remain PNG. Native-image links use the chosen
extension. Retained-data resume requires the same JPEG format and quality; render
reuse is independent of output encoding.

Geometric acceptance is checked before compression. JPEG previews and reported
output scores are regenerated from the saved JPEG, so they include compression
loss; `alignment_audit_before/after` retain the scores before encoding.

## Movement and coverage

The reference is rendered at the selected cameras. RGB, median ray depth and
opacity stay in memory by default and are cached only with
`--keep-alignment-data`; engine allocations are released before inference. Native
photos are prepared through the bounded pipeline. Image buffering is independent
of the number of registered views.

RoMa supplies both correspondence directions, overlap and precision. Confidence,
round-trip consistency, predicted precision and local ZNCC affect anchor weights.
They never directly determine output validity. Small grayscale patches refine
matches to subpixel coordinates. A robust affine movement estimate supplies a
complete baseline; coarse-to-fine, robust grid regularization fits local motion.
Weakly matched areas inherit movement instead of becoming unsupported holes.
`--min-overlap` and `--min-correlation` control soft weighting, `--max-cycle`
controls cycle weighting, and `--max-displacement` excludes implausible anchors.

Reference depth is sampled at matched reference coordinates, not at unwarped
photo coordinates. Depth influences neighbouring motion softly. Surface cuts
require a substantial depth jump and different fitted motion with nearby anchor
support; uncertain geometry alone does not split the source. Cuts are
conservative estimates, not verified physical boundaries. A single visible-depth
map cannot recover all hidden layers or repair incorrect reference geometry.

A source-image triangle mesh covers the entire source pixel domain. Supported
surface boundaries duplicate mesh vertices rather than discarding source area.
Overlapping warped triangles are resolved by depth. Deformation limits reduce
correction locally and refit the mesh; they do not mask problematic triangles.
A photometric line search selects correction strength on fitting patches, keeping
reserved audit patches separate. Insufficient matching or unresolved deformation
yields an unchanged fallback.

The same triangles determine source coordinates and coverage. Final validity is
exactly **warped source coverage intersected with the transported original mask
and alpha**. Neither opacity, confidence, depth uncertainty nor patch score can
remove an otherwise covered pixel. Every excluded native pixel is attributed to
missing coverage or the original source mask. These are exclusions under the
estimated warp, not proof that the warp is geometrically correct.

The final image samples original RGB pixels once, using native-resolution
Lanczos3. Out-of-bounds and excluded filter taps are omitted and remaining weights
normalized; the nearest source pixel determines mask membership. This avoids
blanket mask erosion at borders. No rendered colors or invented texture fill
missing observations. RGB8 PNG or JPEG export uses the normal decoder; embedded
ICC profiles are not copied. Native output size does not imply native matching
precision. Reflections, changing illumination, an inaccurate teacher and very
thin occluders remain difficult.

## Outputs and training

Review files below require `--keep-alignment-data`.

- `index.html`: all selected views on one offline page, with original/render and
  corrected/render blending, synchronized scrolling, zoom, native photograph
  detail, blink, exclusion overlays and expandable matching diagnostics.
- `images/`: native corrected images, preserving subdirectories and replacing
  the source extension with `.png` or `.jpg`. Filename collisions are rejected.
- `masks/`: matching white-usable/black-ignore grayscale PNG sidecars following
  COLMAP's convention: `images/photo.jpg` uses `masks/photo.jpg.png`.
- `manifest.json`: settings, view outcomes, coverage accounting, matching-stage
  counts and before/after correlation and gradient residuals.
- `review/`: render caches, raw bidirectional predictions with checksums,
  movement/confidence/cycle diagnostics, coverage and actual-output previews.
  A rejected candidate is retained separately when quality checks cause fallback.
- `warp.bin`: versioned source mesh and field data; its spatial index is rebuilt
  on loading. Version 1 confidence-masked fields are incompatible.

Use Spirula's **Ignore distractors** mask mode, or the training CLI's
`--apply-loss-for-mask false`. Disable source depth/normal supervision when
training with corrected images.
Re-estimate geometry maps before enabling them, since source maps no longer align.

Fixed patches are excluded from direct anchoring and actual native output is
downscaled for paired before/after scoring. The report includes unavailable patch
counts and coverage separately. Fewer than ten scored patches or no correlation
improvement retains an explicitly labelled unchanged original. These checks
measure agreement with the teacher, not independent 3D reconstruction quality.
Paired retraining and held-out photographs are still needed for that claim.

With `--keep-alignment-data`, `--resume` checks paths, file sizes/mtimes, cameras, selection, settings and
precision, then reuses render and checksummed matching caches. It recomputes the
field, masks and photos; fully cached runs do not load the inference model.
`--reuse-renders PREVIOUS_OUTPUT` seeds a fresh report with compatible renders
after checking provenance. This accepts legacy version-1 render caches, but never
reuses their fields or masks. Stamps detect changes; they are not cryptographic
input provenance. `--render-only` stops after rendering. Repeated corrections
always start from untouched source photographs.

## Verification

`image_alignment_test` checks source-area conservation, foreground/background
motion, known occlusion gaps, zero false holes with missing confidence/depth or
textureless images, source-mask transport without erosion, exact sharp identity
sampling at noninteger native scaling, translation footprint, field and prediction
persistence, scoring and output/symlink protection. `image_alignment_dataset_test`
checks exported native dimensions against original photographs, masks, coverage
accounting, safe paths and the absence of copied camera metadata or point clouds.

The core is portable C++ with no new device kernels. Inference requires
`SS_BUILD_SAM=ON`, independently of the trainer backend.

The manifest stores `max_resolution: 0` for uncapped runs; the viewer labels
these runs **Native**. Resume requires the same resolution policy as the original
run, including an explicit limit when resuming a capped run.

## Comparing working resolutions

Run with `--keep-alignment-data` and the same sample and seed at different `--max-resolution`
values, for example 640, 1280 and 2560. The output dimensions stay native at every
setting. The reference rendering and high-resolution RoMa refinement change;
the coarse matcher stays at width 800. Grid spacing, patch radius and consistency
thresholds use working pixels, so larger resolutions also fit finer local motion
and make those thresholds narrower in native-image coordinates. More working
pixels can improve fine alignment but can also follow noise or inaccurate teacher
geometry. Doubling the longest side produces about four times as many working
pixels; total runtime also includes work that does not scale with this setting.

Combine completed runs without loading the model or running inference:

```sh
spirula align-images --compare HALF_RUN --compare BASE_RUN --compare DOUBLE_RUN \
  --output RESOLUTION_REVIEW
```

The command requires identical source/model provenance, camera parameters,
selection, precision and fitting settings apart from maximum resolution. It
writes a fresh review directory. The page has global resolution checkboxes and
a cycle button, with shared blending, mask overlays, synchronized scrolling and
native-detail inspection within each view. Native images, masks and diagnostics
link to the existing runs, so keep those directories beside the review. Open
`index.html` directly, or serve their common parent directory when using HTTP.

All corrected native photos are downsampled using the same filter and evaluated
against the highest-resolution run's reference render, at common sizes up to 1280
and 2560. Every variant uses the same patch locations and the intersection of
usable support; unavailable or textureless patches are excluded for every
variant together. The small scoring margin around mask boundaries affects only
evaluation, never exported training masks. Correlation and gradient residuals
therefore compare the variants at the same scale, rather than mixing individual
runs' resolution-dependent audit scores. Optional run timing sidecars are shown
with their provenance; an existing cached baseline has no fresh-run timing.
