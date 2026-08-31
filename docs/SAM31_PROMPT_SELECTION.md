# SAM 3.1 prompt selection

Sammie Roto Studio can use the SAM 3.1 multiplex semantic prompt API to find
objects by text before refining them with the existing point workflow.

## Workflow

1. Load footage and set the intended In/Out range.
2. Select **SAM 3.1**, then press **Load Model**.
3. Move to the In frame for forward tracking or the Out frame for backward
   tracking.
4. Enter a short object description such as `person`, `dog`, or `red car`, and
   press **Preview Prompt Candidates**.
5. Inspect the candidates in the viewer and select one or more list entries.
6. Press **Accept Selected**. Each accepted candidate becomes a Studio object
   and receives an interior positive point for normal refinement.
7. Add positive/negative points if needed, then use **Track Objects**.

Prompt preview is non-destructive. It runs in a separate SAM 3.1 session and
does not alter existing points or masks. Accepting a candidate is an explicit
reseed because the official multiplex model resets semantic state when a new
text prompt is applied. If points or generated results already exist, Studio
asks for confirmation before clearing them.

Candidates may be previewed on any frame, but they can only be accepted on the
In or Out frame. This matches SAM 3.1's endpoint-anchor tracking contract.

## Session persistence

The session/project stores the prompt text, anchor frame, and mapping from each
SAM candidate ID to its Studio object ID. On predictor reset or project reopen,
Studio restores the semantic prompt before replaying refinement points.

The standalone points JSON format contains points only. Loading a points JSON
therefore clears semantic prompt metadata; save/open the complete project when
the prompt mapping must be preserved.

The feature requires the official SAM 3.1 multiplex runtime and a compatible
local or Hugging Face checkpoint. It is hidden for the SAM2 and EfficientTAM
backends.
