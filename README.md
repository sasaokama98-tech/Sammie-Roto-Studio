# Sammie Roto Studio
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

Sammie Roto Studio is the VFX-focused extension of [Sammie-Roto 2](https://github.com/Zarxrax/Sammie-Roto-2), adding SAM 3.1 tracking and high-quality trimap/matting workflows while retaining the original application's interaction and export pipeline.

![Sammie Roto Studio screenshot](https://github.com/user-attachments/assets/bc2c99c8-4039-49f1-94ed-65f104a83e8d)

[![GitHub Downloads](https://img.shields.io/github/downloads/Zarxrax/Sammie-Roto-2/total)](https://github.com/Zarxrax/Sammie-Roto-2/releases)
[![GitHub Code License](https://img.shields.io/github/license/Zarxrax/Sammie-Roto-2)](LICENSE)
[![Discord](https://img.shields.io/discord/1437589475369811970?label=Discord&color=blue)](https://discord.gg/jb5qrFyGFF)

## Upstream and acknowledgements

Sammie Roto Studio is an independently maintained downstream extension of [Sammie-Roto 2](https://github.com/Zarxrax/Sammie-Roto-2), created by [Zarxrax](https://github.com/Zarxrax). We sincerely thank Zarxrax and every upstream contributor for the original application, its approachable workflow, and the substantial foundation on which this project is built.

The upstream copyright and the [GNU General Public License v3.0](LICENSE) remain in effect. Studio-specific changes and support requests should be reported to this repository so that the official upstream project is not burdened with issues that only affect this extension.

## Optional MEMatte backend

The high-resolution MEMatte backend loads a user-supplied checkout of the
[official MEMatte repository](https://github.com/linyiheng123/MEMatte) and an
official ViTS checkpoint. Source, checkpoint, memory controls, and current
license limitations are documented in
[`docs/PHASE3_MEMATTE.md`](docs/PHASE3_MEMATTE.md). MEMatte source and weights
are not redistributed with Sammie Roto Studio. Install its Python runtime with
`uv sync --extra <compute-backend> --extra mematte`.

## SAM 3.1 prompt selection

When SAM 3.1 is selected, Studio can preview semantic text-prompt candidates in
an isolated, non-destructive model session. Accepted candidates become normal
Studio objects that can be refined with positive/negative points and tracked
forward from In or backward from Out. The prompt, anchor, and object mappings
are retained in project settings. See
[`docs/SAM31_PROMPT_SELECTION.md`](docs/SAM31_PROMPT_SELECTION.md).

## Hybrid HQ

Hybrid HQ runs MatAnyone2 or VideoMaMa as a temporal stage, unloads it, then
uses MEMatte only for uncertain edge ROIs. The stable core/background is
preserved rather than replacing the full alpha. Phase 4.1 adds Preserve
Temporal, Balanced, and Maximum Detail stability presets; Preserve Temporal is
the default and filters only bounded MEMatte residuals across three frames.
An optional Phase 4.2 Motion Confidence mode adds bidirectional DIS alignment
with per-pixel forward/backward fallback; it remains experimental and off by
default. Phase 4.3 can archive named temporal/final/confidence runs and write
no-reference boundary and flow-warped comparison metrics plus a summary CSV.
Controls and processing semantics are documented in
[`docs/PHASE4_HYBRID_HQ.md`](docs/PHASE4_HYBRID_HQ.md).
Memory Safe, Balanced, Fast, and automatically preserved Custom policies are
documented in
[`docs/PHASE5_MEMORY_PROFILES.md`](docs/PHASE5_MEMORY_PROFILES.md).

The integrated installer/updater follows this Studio repository and keeps the
SAM 3.1, ViTMatte, and MEMatte runtime extras installed. Studio version
`2.4.0+studio.2` incorporates the original Sammie-Roto 2 v2.4.0 installer,
device detection, image-loading, progress-feedback, segmentation-preview, and
dependency-configuration updates without replacing the Studio backends or UI.

**Please add a Github Star if you find it useful!**

Sammie-Roto 2 is an easy-to-use, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone), [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), and [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)

Sammie-Roto 2 is free and open source, but runs models produced by several external projects and organizations. Some models may have restrictions on commercial usage. Please check with the relevant model provider if you have questions regarding licensing.

### Updates
**The upstream changelog can be seen under [Sammie-Roto 2 releases](https://github.com/Zarxrax/Sammie-Roto-2/releases).**
- [08/22/2026] 2.4.0 - New installer/updater, additional segmentation tracking options, many small fixes and improvements.
- [04/17/2026] 2.3.3 - Several large performance optimizations, and colorspace conversions are now handled correctly.
- [04/10/2026] 2.3.2 - Improved temporal stability for VideoMaMa.
- [04/02/2026] 2.3.1 - Added a live preview during segmentation while holding the shift key.
- [03/27/2026] 2.3.0 - Added VideoMaMa model, added option to combine objects when matting, fixed major segmentation bug, and more.
- [03/08/2026] 2.2.0 - Added MatAnyone2 model.
- [01/18/2026] 2.1.1 - Rebuilt the export dialog, slightly faster application startup, bug fixes.
- [12/16/2025] 2.1.0 - Added In/Out markers. Modifying points no longer deletes tracking data. Enabled half-precision for much faster segmentation. Added EfficientTAM model.
- [11/23/2025] 2.0.0 - First stable release. Includes several new features and bugfixes. New quick-start video tutorial and [Discord server](https://discord.gg/jb5qrFyGFF).
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Documentation and Tutorials:
## [Wiki Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

[![Quick Start Video](https://img.youtube.com/vi/m0iZpxsZJcE/0.jpg)](https://www.youtube.com/watch?v=m0iZpxsZJcE)

*This video does not cover new features and changes added since version 2.0

### Installation (Windows):
- Download the latest Studio version from [Sammie Roto Studio releases](https://github.com/sasaokama98-tech/Sammie-Roto-Studio/releases)
- Extract the zip archive to any location that doesn't restrict write permissions (don't put it in Program Files)
- Run 'install.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software (or double click the desktop shortcut).

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder.

### Installation (Linux, Mac)
- Download the latest Studio version from [Sammie Roto Studio releases](https://github.com/sasaokama98-tech/Sammie-Roto-Studio/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command in the terminal: `bash install.sh` then follow the prompt.
- MacOS users: double-click the desktop icon to launch the program. Linux users: `bash run_sammie.sh` or find it in the applications menu.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone) & [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone code)
* [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
* Some icons by [Yusuke Kamiyamane](http://p.yusukekamiyamane.com/)
