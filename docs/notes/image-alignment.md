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
caches, including diagnostic JSON files. Image-warp mode exports no camera
metadata or point clouds. `--resume` and `--render-only` require this option.

## Camera refinement in the fixed world frame

When the misalignment is a camera-pose error, refine the alignment before
considering image deformation:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output ALIGNED \
  --refine-cameras --max-resolution 2560 --all
```

This mode leaves photographs byte-identical and writes a new COLMAP model in
`sparse/0`. Accepted camera poses change; point coordinates, observation tracks,
intrinsics and the model's other metadata files, including georeferencing, are
preserved. The splat itself is never transformed. In world-to-camera convention,
the update is `R_new = delta_R * R_old`,
`t_new = delta_R * t_old + delta_t`; the world's origin, axes and scale stay fixed.

Reliable render-to-photo matches are lifted into 3D using rendered ray depth and
the existing calibration. A robust six-parameter refinement reuses the native
SfM optimizer. Samples need bidirectional overlap, round-trip consistency,
opaque rendered support and locally consistent depth. Depth ranges receive
balanced weights so distant background does not dominate nearby structures.
Reserved correspondences must improve by 10% before a correction is accepted.
The reported pose errors are depth-balanced means with errors clipped at 20
working pixels, not median errors or ground-truth accuracy.

The accepted pose is re-rendered and matched again against the untouched photo.
The review offers initial/refined renders and before/after correspondence-error
maps. Original camera files remain untouched. Only selected images are copied;
unselected camera records remain in the copied model, so a sampled run is a
review subset rather than a complete replacement image tree.

Camera refinement currently requires COLMAP without rig frames and a fresh,
complete run (no `--resume` or `--render-only`). It runs serially so rendering
and matching do not compete for the GPU. Source filenames
and encodings are preserved; image-encoding options do not recompress them.
For small remaining image errors, a separate image-warp run can use this refined
dataset and keep its camera poses fixed.

Camera refinement processes one image through both stages before moving to the
next. By default, dense renders, depths, RoMa predictions and warp fields stay
in RAM and are released after that image; no dense temporary files are written,
including under `/tmp`. Sparse observation coordinates are transported while the
field is still in memory. Only compact camera/observation metadata accumulates
until the COLMAP model is written. The output contains only `images/`, `masks/`
and `sparse/`.

`--keep-alignment-data` explicitly enables the review and persistent caches. It
can retain hundreds of megabytes per view at a 2560-pixel working resolution;
use it for small review samples rather than routine full-dataset processing.

To apply both stages in one run:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output CORRECTED \
  --refine-cameras --residual-warp --max-resolution 2560 --all
```

The residual fit uses fresh matches against the accepted camera's render. A
rejected pose retains the original camera and still permits residual correction.
The residual stage has its own reserved-patch acceptance check; rejecting it
does not undo an accepted pose. Only the final residual field resamples the
original photograph, once, at native resolution. Output encoding follows
`--image-format` and `--jpeg-quality`; with `--keep-alignment-data`, original
native photos are retained in the review. Before/after error maps evaluate the residual stage against
the accepted pose, and the initial render remains available separately.

The exported COLMAP model uses the output filenames and transports selected
2D observation coordinates through the same mesh, including the half-pixel
conversion between COLMAP's corner convention and the image field's integer
pixel centers. Observation indices and 3D track associations remain intact.
Intrinsics, 3D points and georeferencing remain unchanged. Stored sparse-point
reprojection-error summaries describe the input reconstruction; they are not
recomputed by this preprocessing command. This output is not a new bundle
adjustment or an independent validation of the sparse geometry.

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
complete baseline. A robust data fit penalizes residual motion and bending on
the grid. Large correspondence residuals lose weight rather than weakening
smoothness wherever fitted motions disagree. Coherent grayscale gradients in
the source photograph strengthen bending constraints along their tangent,
over spans of 2, 4, 8 and 16 grid intervals. Long constraints require continuous
edge evidence across the span; incoherent texture and corners weaken this
constraint. These are soft constraints, not a guarantee of exact straightness.
Weakly matched areas inherit movement instead of becoming unsupported holes.
`--min-overlap` and `--min-correlation` control soft weighting, `--max-cycle`
controls cycle weighting, and `--max-displacement` excludes implausible anchors.
Its default, `0`, resolves to 5% of the working image diagonal. A positive value
is an explicit limit in working pixels. The automatic limit scales with
resolution so that increasing matching resolution does not reject the same
physical movement. Resume of a run made with the old explicit 40-pixel default
requires `--max-displacement 40`.

Reference depth and round-trip consistency are evaluated again after patch
refinement, at the final matched coordinates. Depth influences neighbouring
motion softly only when at least three other nearby anchors support each
sample's depth. Surface cuts require this support on both sides, a substantial
depth jump and different fitted motion; an isolated depth outlier cannot borrow
confidence from unrelated neighbouring surfaces to split the source. Cuts are
conservative estimates, not verified physical boundaries. A single visible-depth
map cannot recover all hidden layers or repair incorrect reference geometry.

A source-image triangle mesh covers the entire source pixel domain. Supported
surface boundaries duplicate mesh vertices rather than discarding source area.
Overlapping warped triangles are resolved by depth. Deformation limits reduce
local residuals toward a conservative affine transform of each surface; they
do not pull an aligned surface back toward zero movement or mask problematic
triangles. A photometric line search selects residual strength relative to those
surface transforms on fitting patches, keeping
reserved audit patches separate. Insufficient matching or unresolved deformation
yields an unchanged fallback.
The reported `correction_scale` is the minimum residual retention, not a scale
on the complete camera or surface movement.

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

- `index.html`: one image pane per selected view, with a dropdown for original,
  corrected, render, training-mask and warp-coverage pairs. Blend, draggable wipe
  and RGB difference modes share zoom and native-detail controls. Top controls
  update every view; individual controls override them. Resolution comparisons
  also offer direct corrected/corrected pairs. Exclusion overlays and expandable
  matching diagnostics use existing images; changing the viewer rerenders no splats.
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
- `alignment-error-before.png` / `alignment-error-after.png`: discrepancy from
  bidirectionally consistent RoMa correspondences on the same working-pixel
  scale (green 0, red 10 or more). Gray is unverified matching and magenta is
  missing warped coverage. These measure agreement with the matcher, not ground
  truth. Use them alongside render overlays and actual photographic edges.

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
persistence, scoring and output/symlink protection.
`image_alignment_structure_test` checks straight vertical and diagonal beams
under cycle-consistent correspondence noise at several wavelengths, including
isolated depth outliers, retained translation and interior coverage.
It also checks correction of synthetic radial distortion, requiring subpixel
mean edge error and at least 80% reduction of the original edge displacement.
`image_alignment_pose_test` checks a calibrated two-depth scene with large
foreground parallax, independent background motion and held-out pose residuals.
`image_alignment_dataset_test`
checks exported native dimensions against original photographs, masks, coverage
accounting and safe paths. Image-only exports exclude camera metadata; camera
refinement exports require the COLMAP model. Passing a retained-data run as its
third argument also checks that a compact run contains only final dataset files
and produces byte-identical photographs, masks and sparse metadata.

The core is portable C++ with no new device kernels. Inference requires
`SS_BUILD_SAM=ON`, independently of the trainer backend.

## Registration research

- [GS-CPR](https://arxiv.org/abs/2408.11085) establishes 2D matches against
  splat renders, lifts them with rendered depth, then estimates camera pose.
- [iComMa](https://arxiv.org/abs/2312.09031) combines matching and rendered-image
  comparison for pose estimation; their capture ranges and precision differ.
- [UGS-Loc](https://arxiv.org/abs/2603.16538) models pose and geometric
  uncertainty. Plausible splat appearance does not make every depth trustworthy.
- [RoMa v2](https://arxiv.org/abs/2511.15706) separates coarse matching from
  high-resolution refinement and predicts directional covariance for estimation.
- [MoGe-2](https://arxiv.org/abs/2507.02546) predicts detailed monocular geometry
  and metric scale. Such predictions could supply an additional boundary or
  depth-consistency cue, but would need validation in the splat's coordinate
  frame; the alignment command does not currently run an extra depth model.
- [Self-Calibrating Gaussian Splatting](https://arxiv.org/abs/2502.09563)
  jointly optimizes lens distortion, camera parameters and splats. Its hybrid
  invertible neural/grid field regularizes a flexible distortion model. This
  supports using a smooth residual field, but is a joint reconstruction method,
  not evidence that independent per-image warps estimate physical lens parameters.
- [Geometric Structure Preserving Warp](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Geometric_Structure_Preserving_Warp_for_Natural_Image_Stitching_CVPR_2022_paper.html)
  adds geometric structure constraints to local alignment. The alignment mesh
  here uses photographic ridge tangents and bending penalties rather than that
  paper's full stitching formulation.
- [COLMAP camera models](https://colmap.github.io/cameras.html) distinguish
  pinhole cameras, which assume prior undistortion, from radial and tangential
  lens models. [Shared intrinsics](https://colmap.github.io/faq.html#share-intrinsics)
  allow photographs from the same camera to constrain a common calibration.

Pose and residual image correction solve different parts of the mismatch. The
residual field may absorb remaining lens distortion, rolling-shutter effects or
reference-geometry errors; matching one image to a splat cannot identify their
causes. A physical lens-calibration extension should fit a shared model across
views with the same optics, validate on held-out views, and retain local image
corrections only where that model is insufficient. This command does not change
the shared intrinsic calibration or claim to recover a calibrated lens model.

World-frame preservation is not sparse-reprojection preservation. A pose that
agrees better with the reference splat can disagree substantially with the
original SfM points if the two representations differ locally. Evaluate both
render agreement and reprojection against the original sparse observations.
Transporting observations through a residual warp preserves their pixel
locations in the delivered images; it cannot reconcile inconsistent 3D geometry.
That requires reference-aware multiview refinement, with the world frame anchored
and validation on additional photographs, rather than moving the global splat
or silently treating its rendered depth as ground truth.

The camera refinement here uses local depth checks and robust refinement. It does not
implement UGS-Loc's Fisher-information fields or Monte Carlo pose sampling.

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
writes a fresh review directory. Global resolution checkboxes and a cycle button
select the corrections offered in each view's pair dropdown, including direct
comparisons between correction resolutions. Blend, wipe, RGB difference, mask
overlays and native-detail inspection share one pane per view. Native images, masks and diagnostics
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
