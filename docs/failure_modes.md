# Failure mode taxonomy

Every unsuccessful trial is assigned exactly one of the following. The taxonomy is
fixed before data collection so that categories cannot be adjusted to fit results.

| Category | Boundary |
|---|---|
| segmentation | the object mask is wrong or missing |
| pose / axis estimation | mask is correct, estimated pose is not |
| planning | pose is correct, no valid motion plan is found |
| approach collision | plan exists, contact occurs before the grasp pose is reached |
| failed closure | gripper reaches the pose but does not close on the object |
| slip during lift | object is held, then lost during the lift |
| wrong functional region | object is lifted stably, but gripped outside the task-relevant region |

The first two are caught by the quality gate and do not consume a grasp trial.
The last is the failure this line of work targets: a grasp can be mechanically
perfect and still functionally useless.
