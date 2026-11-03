# Splat-guided photograph alignment

`spirula align-images` is a separate preprocessing command using the existing
dataset readers, splat renderer, native RoMa v2 and image I/O. It does not train
Gaussians or modify the editor. The source dataset is read-only; output must be
a fresh directory outside it, including through symlink aliases.

The default refines camera poses and then corrects residual image distortion.
No alignment-mode flags are needed. `--camera-only` changes poses without
warping photographs; `--warp-only` keeps the input poses fixed and supports
datasets other than COLMAP. `--refine-cameras` and `--residual-warp` remain
accepted for existing commands, but are redundant in the default mode.

```sh
spirula align-images --data DATASET --reference MODEL.ply --output REVIEW \
  --checkpoint CHECKPOINT.pt --sample 10 --seed 20261031 --review
```

The default is a uniform sample of ten registered images without replacement.
The selected source views are never replaced with easier photographs. With
`--review`, `selection.json` records them before rendering and the
review includes failures and copies of sampled originals for native inspection.

Full conversion uses the same path and requires explicit selection:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output CORRECTED \
  --checkpoint CHECKPOINT.pt --all
```

No full-dataset conversion is needed for evaluation. Without diagnostic retention,
no extra copies of original photographs are written. Without `--max-resolution`, rendering and fine matching
use each photograph’s native dimensions with no resolution cap. Providing
`--max-resolution PIXELS` explicitly caps the longest rendering/matching side;
delivered images retain native dimensions in both cases. `--primitive
3dgs|mip|3dgut` selects the renderer. The reference must share the dataset's
coordinate frame. Placement and appearance corrections are not inferred from
a loose PLY. Exact supported camera models are retained; approximate lens fits
are refused before output creation. Matching defaults to FP32; `--precision
mixed|auto` selects the existing inference alternatives.

## Minimal training output

The default writes corrected `images/`, lossless `masks/` and the updated COLMAP
model in `sparse/0`. No alignment caches, warp fields, diagnostic images, review
page or extra original photographs are generated. `--warp-only` writes just
images and masks; that mode uses the original dataset's camera alignment and
point cloud for training.

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

In image-only mode (`--warp-only`), on three seeded CinqueTorri views with 1280-pixel working resolution, native JPEG
quality 95 output, eight OpenMP threads and an RTX 3090, elapsed time was 40.24 s
serial and 32.46 s pipelined. All images and masks were byte-identical.
This small comparison includes startup and is not a throughput
forecast for native-resolution matching or a full dataset.

`--reuse-renders RUN`
can read compatible renders and matches directly from an existing alignment
folder without copying those caches into the output.

Add `--review` for `index.html`, original/corrected/render comparisons, native
detail, masks, correspondence-error maps and per-view methods and validation
scores. It writes previews and diagnostic JSON, without dense render, matching
or warp caches. It does retain native originals for inspection. This is an
explicit review output; leave the flag off for minimal training output.

`--keep-alignment-data` includes the review and additionally retains reusable
binary caches and warp fields. Image-warp mode exports no camera metadata or
point clouds. `--resume` requires this cache option and `--warp-only`.
`--render-only --warp-only` accepts either retention option; `--review` saves
only render previews in that mode.

For lightweight numerical investigation, `--metrics PATH` writes a compact JSON
report outside both the input and output datasets, without retaining dense
intermediates or previews. It records camera acceptance reasons, proposed pose
errors, image agreement and residual-correction outcomes. This is opt-in; agent
test reports belong under `/tmp`.

## Camera refinement in the fixed world frame

When the misalignment is a camera-pose error, refine the alignment before
considering image deformation:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output ALIGNED \
  --camera-only --max-resolution 2560 --all
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
The original RoMa correspondences propose a rigid pose. Low-pass image patches
at quarter, half and full working resolution refine those correspondences for
another rigid proposal and a rotation-only proposal. The latter is further
optimized directly against low-pass patch agreement with held-out patches. It
fixes camera position when rendered depth does not support reliable translation.
These are camera transforms, not image deformations.

A direct six-parameter image refinement complements the rotation-only proposal.
It normalizes scene range internally so numerical translation derivatives remain
useful regardless of world-coordinate units. The exported translation remains in
the original world units. Foreground gradient-profile matches also supply
normal-only camera constraints, with depth taken from nearby opaque, locally
stable rendered support. Their weights are reduced if the fit damages the
held-out interior matches. Separate edge locations are reserved for validation.

Every candidate is rendered from the splat and compared with the photograph on
common patches at fine and coarse scales. Image validation requires mean NCC
gain above 0.001 without degrading either scale by more than 0.001, with at least
50 scored patches per scale. If none passes, the geometric proposal is matched
again with RoMa. It can pass through reduced bidirectional displacement at the
same source-photo locations, even when render appearance disagrees with pixels.
This requires 128 common locations, at least half the original support in common,
and at least 80% as many reliable matches overall. Mean displacement, clipped at
20 working pixels, must fall by 10% and 0.1 working pixels. Lost original matches
keep their original error and contribute no improvement. That score over all
original support must also improve by 10%, preventing a better score merely
from discarding difficult locations. The selected fresh matches are reused for
residual correction. No additional dense data is written.

An edge-guided proposal can also pass through fresh gradient-profile matching
on its actual render: at least 12 common held-out edges, 70% of the original
edge support in common, 80% of the original total support, and a 20% and
0.25-pixel reduction in normal displacement. Missing original matches earn no
improvement. Fine and coarse image agreement may each lose at most 0.01 NCC
through this route. Image-validated proposals retain their existing validation.
An added direct or edge-guided proposal must improve mean image agreement by
another 0.001 NCC to replace an already accepted proposal. Marginal score
differences do not displace a camera that already validated.

When a corrected camera still has a substantial structural residual, one more
edge-guided pose step is allowed. It reuses the already computed RoMa matches
to propose the update, then re-renders and validates before another neural
matching pass. Rejected trials retain the accepted pose and its matches. This
is bounded to one additional step; ordinary views do not enter an open-ended
render/match loop. Review metadata records candidates and accepted extra steps.

A reduction of the original fitting residual alone does not establish that the
new render aligns better. The reported pose errors
are depth-balanced means clipped at 20 working pixels, not median errors or
ground-truth accuracy. Keeping the input pose does not disable residual image
correction in the default mode.

With too few usable geometry matches, the command can test up to four distinct
nearby registered poses in filename order, scanning at most 64 entries in each
direction. It tries learned matching from those renders, allowing wider initial
displacements when the user has not set an explicit displacement limit. The
candidate is re-rendered and re-matched; one additional pose refinement can
polish it. Recovery requires at least 128 geometry matches, ten held-out matches,
an error below three working pixels in the final matching and direct fine/coarse
image agreement. These
are alternative initializations, not unconditional copying of a neighbour's
camera. Ordered capture names help this bounded search; unrelated filename
ordering does not provide the same initialization prior.

The accepted pose is re-rendered and matched again against the untouched photo.
The review offers initial/refined renders and before/after correspondence-error
maps. Original camera files remain untouched. Only selected images are copied;
unselected camera records remain in the copied model, so a sampled run is a
review subset rather than a complete replacement image tree.

Camera refinement currently requires COLMAP without rig frames and a fresh,
complete run (no `--resume` or `--render-only`). Rendering and matching share one
GPU worker; CPU residual fitting, native resampling and encoding of the previous
view overlap with the next view's GPU work. `--no-pipeline` selects serial
execution. Source filenames
and encodings are preserved; image-encoding options do not recompress them.
For small remaining image errors, a separate image-warp run can use this refined
dataset and keep its camera poses fixed.

Camera refinement keeps a bounded queue of GPU work, CPU output and input
prefetch. By default, dense renders, depths, RoMa predictions and warp fields stay
in RAM and are released after their image; no dense temporary files are written,
including under `/tmp`. Sparse observation coordinates are transported while the
field is still in memory. Only compact camera/observation metadata accumulates
until the COLMAP model is written. The output contains only `images/`, `masks/`
and `sparse/`.

`--keep-alignment-data` explicitly enables the review and persistent caches. It
can retain hundreds of megabytes per view at a 2560-pixel working resolution;
use it for small review samples rather than routine full-dataset processing.
Agent validation runs belong under `/tmp`; production commands must not enable
diagnostic retention unless the user explicitly requests it. See the production
output rules in `AGENTS.md`.

Both stages run by default:

```sh
spirula align-images --data DATASET --reference MODEL.ply --output CORRECTED \
  --max-resolution 2560 --all
```

The residual fit uses fresh matches against the accepted camera's render. A
rejected pose retains the original camera and still permits residual correction.
The residual stage has its own reserved-patch acceptance check; rejecting it
does not undo an accepted pose. Only the final residual field resamples the
original photograph, once, at native resolution. Output encoding follows
`--image-format` and `--jpeg-quality`; with `--review`, original
native photos are retained in the review. Before/after error maps evaluate the residual stage against
the accepted pose, and the initial render remains available separately.

The exported COLMAP model uses the output filenames and transports selected
2D observation coordinates through the same mesh, including the half-pixel
conversion between COLMAP's corner convention and the image field's integer
pixel centers. Observation indices and 3D track associations remain intact.
Existing observations outside the source photograph are retained unchanged;
there is no photographic warp to evaluate there. In-image observations may move
outside the output frame. Neither case rejects or removes the photograph.
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

Thin foreground edges also supply independent normal-direction matches. The
source and render are low-pass filtered before comparing signed gradient profiles
across each coherent source edge. A smaller tensor window locates thin ridges
inside a mesh cell when the larger window is dominated by background texture.
A search within 16 working pixels of RoMa needs
profile agreement above 0.65, a distinct peak separated by 0.15 from alternatives,
and a reverse match within one pixel. Positive foreground depth with at least
half rendered coverage must be available near the matched edge; its surroundings
must be substantially farther
away or lack depth. Missing background depth does not invalidate a visible
foreground edge. Source masks cover the complete profile footprint. Nearby
parallel matches must agree, and a local median rejects small groups that lock
onto a neighboring parallel part.

These correspondences constrain only displacement perpendicular to the edge in
the existing mesh solver. They do not infer motion along a featureless line or
increase global smoothing. Near accepted foreground constraints, fine-scale
source ridges propagate motion along poorly matched extensions, so an unsupported
tip does not bend toward background motion. This propagation is limited to
weakly supported correspondence regions. Existing deformation limits, one-pass native sampling
and reserved-patch acceptance still apply. This is a local image alignment cue,
not a physical lens calibration; camera intrinsics remain unchanged. If these
additional constraints cause unresolved deformation or fail the image audit, the
same fitter retries with the original RoMa constraints before retaining a source
photograph. A structural fit also cannot promote a previously rejected warp when
the outer 10% of the image loses more than 0.02 NCC over at least 100 audit patches.
This promotion check preserves a conservative fallback in poorly rendered views;
it does not veto an already usable RoMa correction or a validated camera pose.
Both candidates sample the original pixels; resampling is not chained.

Straight segments detected independently in the photograph protect against a
misshapen reference. Signed gradients must trace the same straight edge through
nine samples, over 32- and 64-pixel spans. Arbitrary segment directions contribute
normal-only second-difference constraints through bilinear mesh interpolation.
A 0.25-pixel curvature allowance preserves small smooth corrections; larger local
bends are penalized even if reference matching prefers them. These constraints
also respect source masks. The conservative retry disables both the added edge
matches and the source-line constraints, preserving the original RoMa fallback.

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

An exception during correction retains the original photograph and source mask,
with an already validated camera correction if available. The terminal identifies
the unchanged photo and reason. Matching or metadata errors do not remove views
from the training dataset. An unreadable source or unwritable output can still
prevent export and is reported as a failed write. Dense diagnostics remain opt-in.

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

Review files below require `--review`, which is also implied by
`--keep-alignment-data`. Training images and masks are always written.

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
- `review/`: movement/confidence/cycle diagnostics, coverage and actual-output
  previews. Render caches and raw bidirectional predictions with checksums are
  included only with `--keep-alignment-data`.
  A rejected candidate is retained separately when quality checks cause fallback.
- `warp.bin` (only with `--keep-alignment-data`): versioned source mesh and field
  data; its spatial index is rebuilt on loading. Version 1 confidence-masked
  fields are incompatible.
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

Use fixed camera poses (`--warp-only`) for resolution comparisons. Run with
`--review` and the same sample and seed at different `--max-resolution`
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
