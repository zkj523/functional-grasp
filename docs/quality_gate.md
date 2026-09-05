# Pre-execution quality gate

A grasp that fails can fail for two very different reasons: the perception was wrong,
or the execution was. Without separating them, a table of success rates carries no
diagnostic information — and worse, a badly estimated pose can drive the arm into the
table.

The gate is evaluated after perception and **before any motion command is issued**.
If any check fails the run aborts and is recorded as a perception failure; it never
enters the grasp statistics.

| Check | What it catches |
|---|---|
| registration p95 / mean residual | pose estimate that does not explain the observed points |
| table plane fit RMSE | bad segmentation, or the object is not resting on the fitted plane |
| visible object point count | severe occlusion, or the object left the field of view |
| observed axis span vs nominal | the object is only partially visible; the pose is under-constrained |
| approach-direction alignment | the planned approach is inconsistent with the estimated pose |

Two properties are deliberate:

**The gate is a pure predicate.** `validate_online_chips_can_run.py` reads recorded
numbers and returns a verdict. It has no side effects and does not touch the robot,
so it can be re-run on an archived record to reproduce the original decision.

**The thresholds are recorded, not just the verdict.** `quality_gate.json` stores
every metric together with the threshold it was compared against, so a later reader
can tell whether a run passed comfortably or marginally.
